"""
Multi-seed validation for statistical robustness
Runs experiments with 5 different seeds and reports mean ± std
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score

warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ============== CONFIGURATION ==============

SEEDS = [42, 123, 456, 789, 1011]

DATASETS = {
    'Caco2_Wang': 'regression',
    'Half_Life_Obach': 'regression',
    'Clearance_Hepatocyte_AZ': 'regression',
    'Clearance_Microsome_AZ': 'regression',
    'tox21': 'classification',
    'herg': 'classification',
}

# Best hyperparameters from HPO runs (use your actual best params)
BEST_PARAMS = {
    'Caco2_Wang': {'hidden_dim': 256, 'num_layers': 4, 'lr': 0.001, 'dropout': 0.2},
    'Half_Life_Obach': {'hidden_dim': 256, 'num_layers': 4, 'lr': 0.001, 'dropout': 0.2},
    'Clearance_Hepatocyte_AZ': {'hidden_dim': 256, 'num_layers': 4, 'lr': 0.001, 'dropout': 0.2},
    'Clearance_Microsome_AZ': {'hidden_dim': 256, 'num_layers': 4, 'lr': 0.001, 'dropout': 0.2},
    'tox21': {'hidden_dim': 256, 'num_layers': 4, 'lr': 0.001, 'dropout': 0.2},
    'herg': {'hidden_dim': 256, 'num_layers': 4, 'lr': 0.001, 'dropout': 0.2},
}

MAX_EPOCHS = 50
PATIENCE = 12
OUTPUT_DIR = os.path.join(project_root, 'results', 'multi_seed')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataset(name, seed):
    """Load dataset with specific seed"""
    if name.lower() in ['tox21', 'herg']:
        from tdc.single_pred import Tox
        if name.lower() == 'tox21':
            data = Tox(name='tox21', label_name='NR-AR')
        else:
            data = Tox(name='herg')
    else:
        from tdc.single_pred import ADME
        data = ADME(name=name)

    return data.get_split(method='scaffold', seed=seed)


class GNNModel(nn.Module):
    """Simple GCN model"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.lin = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        self.pool = global_mean_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)

        x = self.pool(x, batch)
        return self.lin(x)


def smiles_to_graph(smiles):
    """Convert SMILES to PyG Data"""
    from rdkit import Chem
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atoms = []
    for atom in mol.GetAtoms():
        atoms.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.IsInRing(),
            atom.GetTotalNumHs(),
            atom.GetMass() / 100,
            atom.GetNumRadicalElectrons()
        ])

    x = torch.tensor(atoms, dtype=torch.float)

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([[i, j], [j, i]])

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


def prepare_data(split, batch_size=32, task_type='regression'):
    """Prepare DataLoaders with proper preprocessing

    For regression: applies log transform (if needed) + normalization
    Returns loaders and (mu, sigma) for inverse transform
    """
    from torch_geometric.loader import DataLoader

    graphs = {'train': [], 'valid': [], 'test': []}
    all_ys = {'train': [], 'valid': [], 'test': []}

    for split_name, df in [('train', split['train']), ('valid', split['valid']), ('test', split['test'])]:
        for _, row in df.iterrows():
            g = smiles_to_graph(row['Drug'])
            if g is not None:
                y = row['Y']
                if not np.isnan(y):
                    g.original_y = y
                    all_ys[split_name].append(y)
                    graphs[split_name].append(g)

    # Compute normalization parameters from training data
    y_train = np.array(all_ys['train'])

    if task_type == 'classification':
        mu, sigma = 0.0, 1.0
        for split_name in ['train', 'valid', 'test']:
            for g in graphs[split_name]:
                g.y = torch.tensor([g.original_y], dtype=torch.float)
    else:
        # Regression: detect if values are already log-transformed
        all_negative = np.all(y_train < 0)

        if all_negative:
            # Values already in log space (e.g., Caco2_Wang)
            y_log = y_train.astype(np.float32)
            mu = float(y_log.mean())
            sigma = float(y_log.std())
            if sigma < 1e-6:
                sigma = 1.0
            for split_name in ['train', 'valid', 'test']:
                for g in graphs[split_name]:
                    g.y = torch.tensor([(g.original_y - mu) / sigma], dtype=torch.float)
        else:
            # Positive values: apply log transform then normalize
            clip_min = 1e-6
            y_train_clipped = np.clip(y_train, clip_min, None)
            y_log = np.log(y_train_clipped)
            mu = float(y_log.mean())
            sigma = float(y_log.std())
            if sigma < 1e-6:
                sigma = 1.0
            for split_name in ['train', 'valid', 'test']:
                for g in graphs[split_name]:
                    y_clipped = max(clip_min, g.original_y)
                    g.y = torch.tensor([(np.log(y_clipped) - mu) / sigma], dtype=torch.float)

    train_loader = DataLoader(graphs['train'], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(graphs['valid'], batch_size=batch_size)
    test_loader = DataLoader(graphs['test'], batch_size=batch_size)

    return train_loader, valid_loader, test_loader, (mu, sigma)


def train_and_evaluate(dataset_name, task_type, params, seed, device):
    """Train and evaluate model with given seed"""

    set_seed(seed)

    # Load data with proper preprocessing
    split = get_dataset(dataset_name, seed)
    train_loader, valid_loader, test_loader, (mu, sigma) = prepare_data(split, task_type=task_type)

    # Create model
    model = GNNModel(
        input_dim=9,
        hidden_dim=params['hidden_dim'],
        output_dim=1,
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.BCEWithLogitsLoss() if task_type == 'classification' else nn.MSELoss()

    best_val = float('inf') if task_type == 'regression' else 0
    patience_counter = 0
    best_model_state = None

    # Training
    for epoch in range(MAX_EPOCHS):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out.squeeze(), data.y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for data in valid_loader:
                data = data.to(device)
                out = model(data)
                val_preds.extend(out.cpu().numpy().flatten())
                val_labels.extend(data.y.cpu().numpy().flatten())

        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)

        if task_type == 'classification':
            val_metric = roc_auc_score(val_labels, val_preds)
            improved = val_metric > best_val
        else:
            # Use log-scale RMSE for validation (more stable)
            val_preds_log = val_preds * sigma + mu
            val_labels_log = val_labels * sigma + mu
            val_metric = np.sqrt(mean_squared_error(val_labels_log, val_preds_log))
            improved = val_metric < best_val

        if improved:
            best_val = val_metric
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # Load best model and test
    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            test_preds.extend(out.cpu().numpy().flatten())
            test_labels.extend(data.y.cpu().numpy().flatten())

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)

    if task_type == 'classification':
        test_auc = roc_auc_score(test_labels, test_preds)
        return {
            'seed': seed,
            'best_val_metric': best_val,
            'test_auc': test_auc,
            'test_metric': test_auc  # backward compatibility
        }
    else:
        # Denormalize to log scale
        test_preds_log = test_preds * sigma + mu
        test_labels_log = test_labels * sigma + mu

        # Log-scale metrics (before exp transform)
        rmse_log = np.sqrt(mean_squared_error(test_labels_log, test_preds_log))
        mae_log = mean_absolute_error(test_labels_log, test_preds_log)

        # Original-scale metrics (after exp transform)
        test_preds_orig = np.exp(test_preds_log)
        test_labels_orig = np.exp(test_labels_log)
        rmse_orig = np.sqrt(mean_squared_error(test_labels_orig, test_preds_orig))

        return {
            'seed': seed,
            'best_val_metric': best_val,
            'test_rmse_orig': rmse_orig,
            'test_rmse_log': rmse_log,
            'test_mae_log': mae_log,
            'test_metric': rmse_log  # backward compatibility (use log scale)
        }


def compute_statistics(results, task_type):
    """Compute mean, std, CI for all metrics"""

    def calc_stats(values):
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        n = len(values)
        ci = stats.t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n))
        return {
            'mean': mean,
            'std': std,
            'min': np.min(values),
            'max': np.max(values),
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'values': values
        }

    if task_type == 'classification':
        auc_values = [r['test_auc'] for r in results]
        return {
            'auc': calc_stats(auc_values),
            'test_metric': calc_stats(auc_values)  # backward compatibility
        }
    else:
        rmse_orig_values = [r['test_rmse_orig'] for r in results]
        rmse_log_values = [r['test_rmse_log'] for r in results]
        mae_log_values = [r['test_mae_log'] for r in results]
        return {
            'rmse_orig': calc_stats(rmse_orig_values),
            'rmse_log': calc_stats(rmse_log_values),
            'mae_log': calc_stats(mae_log_values),
            'test_metric': calc_stats(rmse_log_values)  # backward compatibility
        }


def run_multi_seed_benchmark():
    """Run multi-seed benchmark"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print("MULTI-SEED VALIDATION")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"{'='*70}\n")

    all_results = {}

    for dataset_name, task_type in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name} ({task_type})")
        print(f"{'='*60}")

        params = BEST_PARAMS.get(dataset_name, BEST_PARAMS['Caco2_Wang'])
        seed_results = []

        for seed in SEEDS:
            print(f"  Seed {seed}...", end=' ')
            try:
                result = train_and_evaluate(dataset_name, task_type, params, seed, device)
                seed_results.append(result)
                if task_type == 'classification':
                    print(f"AUC={result['test_auc']:.4f}")
                else:
                    print(f"RMSE_log={result['test_rmse_log']:.4f}, MAE_log={result['test_mae_log']:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")

        # Compute statistics
        stats_result = compute_statistics(seed_results, task_type)
        all_results[dataset_name] = {'task_type': task_type, **stats_result}

        if task_type == 'classification':
            s = stats_result['auc']
            print(f"\n  Summary (AUC): {s['mean']:.4f} +/- {s['std']:.4f}")
            print(f"  95% CI: [{s['ci_lower']:.4f}, {s['ci_upper']:.4f}]")
        else:
            s_orig = stats_result['rmse_orig']
            s_log = stats_result['rmse_log']
            s_mae = stats_result['mae_log']
            print(f"\n  RMSE (orig): {s_orig['mean']:.4f} +/- {s_orig['std']:.4f}")
            print(f"  RMSE (log):  {s_log['mean']:.4f} +/- {s_log['std']:.4f}")
            print(f"  MAE (log):   {s_mae['mean']:.4f} +/- {s_mae['std']:.4f}")
            print(f"  95% CI (log): [{s_log['ci_lower']:.4f}, {s_log['ci_upper']:.4f}]")

    # Save results
    save_results(all_results)

    return all_results


def save_results(results):
    """Save results to files with all metrics"""

    # JSON
    with open(f"{OUTPUT_DIR}/multi_seed_results.json", 'w') as f:
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump(results, f, indent=2, default=convert)

    # CSV summary with all metrics
    rows = []
    for dataset, data in results.items():
        task_type = data.get('task_type', 'regression')
        if task_type == 'classification':
            s = data['auc']
            rows.append({
                'Dataset': dataset,
                'Task': task_type,
                'RMSE_orig_Mean': None, 'RMSE_orig_Std': None,
                'RMSE_log_Mean': None, 'RMSE_log_Std': None,
                'MAE_log_Mean': None, 'MAE_log_Std': None,
                'AUC_Mean': s['mean'], 'AUC_Std': s['std'],
                'CI_Lower': s['ci_lower'], 'CI_Upper': s['ci_upper']
            })
        else:
            s_orig = data['rmse_orig']
            s_log = data['rmse_log']
            s_mae = data['mae_log']
            rows.append({
                'Dataset': dataset,
                'Task': task_type,
                'RMSE_orig_Mean': s_orig['mean'], 'RMSE_orig_Std': s_orig['std'],
                'RMSE_log_Mean': s_log['mean'], 'RMSE_log_Std': s_log['std'],
                'MAE_log_Mean': s_mae['mean'], 'MAE_log_Std': s_mae['std'],
                'AUC_Mean': None, 'AUC_Std': None,
                'CI_Lower': s_log['ci_lower'], 'CI_Upper': s_log['ci_upper']
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/multi_seed_summary.csv", index=False)

    # LaTeX table with all metrics
    latex = r"""
\begin{table}[htbp]
\caption{Multi-seed validation results (mean $\pm$ std, n=5 seeds)}
\label{tab:multi_seed}
\centering
\begin{tabular}{lcccc}
\toprule
\textbf{Dataset} & \textbf{RMSE (orig)} & \textbf{RMSE (log)} & \textbf{MAE (log)} & \textbf{AUC} \\
\midrule
"""
    for dataset, data in results.items():
        task_type = data.get('task_type', 'regression')
        if task_type == 'classification':
            s = data['auc']
            latex += f"{dataset} & - & - & - & {s['mean']:.4f} $\\pm$ {s['std']:.4f} \\\\\n"
        else:
            s_orig = data['rmse_orig']
            s_log = data['rmse_log']
            s_mae = data['mae_log']
            latex += f"{dataset} & {s_orig['mean']:.4f} $\\pm$ {s_orig['std']:.4f} & {s_log['mean']:.4f} $\\pm$ {s_log['std']:.4f} & {s_mae['mean']:.4f} $\\pm$ {s_mae['std']:.4f} & - \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    with open(f"{OUTPUT_DIR}/multi_seed_table.tex", 'w') as f:
        f.write(latex)

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_multi_seed_benchmark()
