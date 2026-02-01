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
from sklearn.metrics import mean_squared_error, roc_auc_score

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


def prepare_data(split, batch_size=32):
    """Prepare DataLoaders"""
    from torch_geometric.loader import DataLoader

    graphs = {'train': [], 'valid': [], 'test': []}

    for split_name, df in [('train', split['train']), ('valid', split['valid']), ('test', split['test'])]:
        for _, row in df.iterrows():
            g = smiles_to_graph(row['Drug'])
            if g is not None:
                y = row['Y']
                if not np.isnan(y):
                    g.y = torch.tensor([y], dtype=torch.float)
                    graphs[split_name].append(g)

    train_loader = DataLoader(graphs['train'], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(graphs['valid'], batch_size=batch_size)
    test_loader = DataLoader(graphs['test'], batch_size=batch_size)

    return train_loader, valid_loader, test_loader


def train_and_evaluate(dataset_name, task_type, params, seed, device):
    """Train and evaluate model with given seed"""

    set_seed(seed)

    # Load data
    split = get_dataset(dataset_name, seed)
    train_loader, valid_loader, test_loader = prepare_data(split)

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
            val_metric = np.sqrt(mean_squared_error(val_labels, val_preds))
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
        test_metric = roc_auc_score(test_labels, test_preds)
    else:
        test_metric = np.sqrt(mean_squared_error(test_labels, test_preds))

    return {
        'seed': seed,
        'best_val_metric': best_val,
        'test_metric': test_metric
    }


def compute_statistics(results):
    """Compute mean, std, CI"""
    values = [r['test_metric'] for r in results]

    mean = np.mean(values)
    std = np.std(values, ddof=1)
    n = len(values)

    # 95% CI
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
                print(f"test={result['test_metric']:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")

        # Compute statistics
        stats_result = compute_statistics(seed_results)
        all_results[dataset_name] = stats_result

        metric_name = 'AUC' if task_type == 'classification' else 'RMSE'
        print(f"\n  Summary: {stats_result['mean']:.4f} +/- {stats_result['std']:.4f} ({metric_name})")
        print(f"  95% CI: [{stats_result['ci_lower']:.4f}, {stats_result['ci_upper']:.4f}]")

    # Save results
    save_results(all_results)

    return all_results


def save_results(results):
    """Save results to files"""

    # JSON
    with open(f"{OUTPUT_DIR}/multi_seed_results.json", 'w') as f:
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump(results, f, indent=2, default=convert)

    # CSV summary
    rows = []
    for dataset, stats in results.items():
        rows.append({
            'Dataset': dataset,
            'Mean': stats['mean'],
            'Std': stats['std'],
            'CI_Lower': stats['ci_lower'],
            'CI_Upper': stats['ci_upper'],
            'Min': stats['min'],
            'Max': stats['max'],
            'Result': f"{stats['mean']:.4f} +/- {stats['std']:.4f}"
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/multi_seed_summary.csv", index=False)

    # LaTeX table
    latex = r"""
\begin{table}[htbp]
\caption{Multi-seed validation results (mean $\pm$ std, n=5 seeds)}
\label{tab:multi_seed}
\centering
\begin{tabular}{lccc}
\toprule
\textbf{Dataset} & \textbf{Mean} & \textbf{Std} & \textbf{95\% CI} \\
\midrule
"""
    for row in rows:
        latex += f"{row['Dataset']} & {row['Mean']:.4f} & {row['Std']:.4f} & [{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}] \\\\\n"

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
