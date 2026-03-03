"""
TPE (Tree-structured Parzen Estimator) Hyperparameter Optimization
Using Optuna for Bayesian optimization on MANU benchmark
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ============== CONFIGURATION ==============

DATASETS = {
    'Caco2_Wang': 'regression',
    'Half_Life_Obach': 'regression',
    'Clearance_Hepatocyte_AZ': 'regression',
    'Clearance_Microsome_AZ': 'regression',
    'tox21': 'classification',
    'herg': 'classification',
}

N_TRIALS = 50
SEED = 42
MAX_EPOCHS = 50
PATIENCE = 12
OUTPUT_DIR = os.path.join(project_root, 'results', 'tpe_benchmark')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataset(name):
    """Load dataset from TDC"""
    if name.lower() in ['tox21', 'herg']:
        from tdc.single_pred import Tox
        if name.lower() == 'tox21':
            data = Tox(name='tox21', label_name='NR-AR')
        else:
            data = Tox(name='herg')
    else:
        from tdc.single_pred import ADME
        data = ADME(name=name)

    return data.get_split(method='scaffold', seed=SEED)


class GNNModel(nn.Module):
    """Simple GCN model for molecular property prediction"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, task_type):
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
        self.task_type = task_type
        self.pool = global_mean_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)

        x = self.pool(x, batch)
        x = self.lin(x)

        return x


def smiles_to_graph(smiles):
    """Convert SMILES to PyG Data object"""
    from rdkit import Chem
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features
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

    # Edges
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
    """Prepare PyG DataLoaders from TDC split with proper preprocessing

    For regression: applies log transform (if needed) + normalization
    Returns loaders and (mu, sigma) for inverse transform
    """
    from torch_geometric.loader import DataLoader

    graphs = {'train': [], 'valid': [], 'test': []}

    # First pass: collect all values and create graphs
    all_ys = {'train': [], 'valid': [], 'test': []}

    for split_name, df in [('train', split['train']), ('valid', split['valid']), ('test', split['test'])]:
        for _, row in df.iterrows():
            g = smiles_to_graph(row['Drug'])
            if g is not None:
                y = row['Y']
                if not np.isnan(y):
                    g.original_y = y  # Store original value
                    all_ys[split_name].append(y)
                    graphs[split_name].append(g)

    # Compute normalization parameters from training data
    y_train = np.array(all_ys['train'])

    if task_type == 'classification':
        # No transformation for classification
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

            # Normalize directly
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


def train_epoch(model, loader, optimizer, criterion, device, task_type):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        if task_type == 'classification':
            loss = criterion(out.squeeze(), data.y)
        else:
            loss = criterion(out.squeeze(), data.y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, task_type, mu=0.0, sigma=1.0, return_all_metrics=False):
    """Evaluate model with proper inverse transform for regression

    For regression returns dict with:
        - rmse_orig: RMSE on original scale (after exp transform)
        - rmse_log: RMSE on log scale (before exp transform)
        - mae_log: MAE on log scale (TDC leaderboard metric)
    For classification returns AUC

    If return_all_metrics=False (default during HPO), returns single value for optimization.
    """
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            preds.extend(out.cpu().numpy().flatten())
            labels.extend(data.y.cpu().numpy().flatten())

    preds = np.array(preds)
    labels = np.array(labels)

    if task_type == 'classification':
        auc = roc_auc_score(labels, preds)
        if return_all_metrics:
            return {'auc': auc}
        return auc
    else:
        # Denormalize to log scale
        preds_log = preds * sigma + mu
        labels_log = labels * sigma + mu

        # Log-scale metrics (before exp transform)
        rmse_log = np.sqrt(mean_squared_error(labels_log, preds_log))
        mae_log = mean_absolute_error(labels_log, preds_log)

        # Original-scale metrics (after exp transform)
        preds_orig = np.exp(preds_log)
        labels_orig = np.exp(labels_log)
        rmse_orig = np.sqrt(mean_squared_error(labels_orig, preds_orig))

        if return_all_metrics:
            return {
                'rmse_orig': rmse_orig,
                'rmse_log': rmse_log,
                'mae_log': mae_log
            }
        # For HPO optimization, use log-scale RMSE (more stable)
        return rmse_log


class TPEObjective:
    """Objective function for TPE optimization"""

    def __init__(self, dataset_name, task_type, device):
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.device = device

        # Load data once with proper preprocessing
        print(f"  Loading {dataset_name}...")
        split = get_dataset(dataset_name)
        self.train_loader, self.valid_loader, self.test_loader, (self.mu, self.sigma) = prepare_data(split, task_type=task_type)
        print(f"  Train: {len(self.train_loader.dataset)}, Valid: {len(self.valid_loader.dataset)}, Test: {len(self.test_loader.dataset)}")
        if task_type == 'regression':
            print(f"  Normalization: mu={self.mu:.4f}, sigma={self.sigma:.4f}")

    def __call__(self, trial):
        # Suggest hyperparameters
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 384, 512])
        num_layers = trial.suggest_int('num_layers', 2, 8)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)

        # Create model
        input_dim = 9  # Number of atom features
        model = GNNModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            dropout=dropout,
            task_type=self.task_type
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if self.task_type == 'classification':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()

        # Training loop with pruning
        best_val = float('inf') if self.task_type == 'regression' else 0
        patience_counter = 0

        for epoch in range(MAX_EPOCHS):
            train_epoch(model, self.train_loader, optimizer, criterion, self.device, self.task_type)
            val_metric = evaluate(model, self.valid_loader, self.device, self.task_type, self.mu, self.sigma)

            # Report intermediate value
            trial.report(val_metric, epoch)

            # Pruning
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Early stopping
            if self.task_type == 'regression':
                improved = val_metric < best_val
            else:
                improved = val_metric > best_val

            if improved:
                best_val = val_metric
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break

        return best_val


def run_tpe_for_dataset(dataset_name, task_type, device):
    """Run TPE optimization for a single dataset"""

    # Create study
    direction = 'minimize' if task_type == 'regression' else 'maximize'
    sampler = TPESampler(seed=SEED, n_startup_trials=10)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        study_name=f"MANU_{dataset_name}_TPE"
    )

    # Create objective
    objective = TPEObjective(dataset_name, task_type, device)

    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True, n_jobs=1)

    # Evaluate best model on test set
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best params: {study.best_params}")
    print(f"  Best val metric: {study.best_value:.4f}")

    # Get test performance with best params
    best_params = study.best_params
    model = GNNModel(
        input_dim=9,
        hidden_dim=best_params['hidden_dim'],
        output_dim=1,
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout'],
        task_type=task_type
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=best_params['lr'],
                                  weight_decay=best_params['weight_decay'])
    criterion = nn.BCEWithLogitsLoss() if task_type == 'classification' else nn.MSELoss()

    # Retrain with best params
    for _ in range(MAX_EPOCHS):
        train_epoch(model, objective.train_loader, optimizer, criterion, device, task_type)

    test_metrics = evaluate(model, objective.test_loader, device, task_type, objective.mu, objective.sigma, return_all_metrics=True)

    if task_type == 'classification':
        print(f"  Test AUC: {test_metrics['auc']:.4f}")
    else:
        print(f"  Test RMSE (orig): {test_metrics['rmse_orig']:.4f}")
        print(f"  Test RMSE (log):  {test_metrics['rmse_log']:.4f}")
        print(f"  Test MAE (log):   {test_metrics['mae_log']:.4f}")

    result = {
        'dataset': dataset_name,
        'algorithm': 'TPE',
        'task_type': task_type,
        'best_params': best_params,
        'best_val_metric': study.best_value,
        'n_trials': len(study.trials),
        'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'optimization_history': [
            {'trial': t.number, 'value': t.value, 'params': t.params}
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
    }

    # Add test metrics
    if task_type == 'classification':
        result['test_auc'] = test_metrics['auc']
        result['test_metric'] = test_metrics['auc']  # backward compatibility
    else:
        result['test_rmse_orig'] = test_metrics['rmse_orig']
        result['test_rmse_log'] = test_metrics['rmse_log']
        result['test_mae_log'] = test_metrics['mae_log']
        result['test_metric'] = test_metrics['rmse_log']  # backward compatibility

    return result


def run_tpe_benchmark():
    """Run TPE benchmark on all datasets"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print("TPE (BAYESIAN) HPO BENCHMARK")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Trials per dataset: {N_TRIALS}")
    print(f"Seed: {SEED}")
    print(f"{'='*70}\n")

    set_seed(SEED)

    all_results = []

    for dataset_name, task_type in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name} ({task_type})")
        print(f"{'='*60}")

        try:
            result = run_tpe_for_dataset(dataset_name, task_type, device)
            all_results.append(result)

            # Save individual result
            with open(f"{OUTPUT_DIR}/tpe_{dataset_name}_results.json", 'w') as f:
                json.dump(result, f, indent=2, default=str)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save summary with all metrics
    summary_rows = []
    for r in all_results:
        row = {
            'dataset': r['dataset'],
            'algorithm': 'TPE',
            'task_type': r['task_type'],
            'hidden_dim': r['best_params']['hidden_dim'],
            'num_layers': r['best_params']['num_layers'],
            'lr': r['best_params']['lr'],
            'dropout': r['best_params']['dropout'],
            'n_trials': r['n_trials'],
            'n_pruned': r['n_pruned']
        }
        if r['task_type'] == 'classification':
            row['test_auc'] = r['test_auc']
            row['test_rmse_orig'] = None
            row['test_rmse_log'] = None
            row['test_mae_log'] = None
        else:
            row['test_auc'] = None
            row['test_rmse_orig'] = r['test_rmse_orig']
            row['test_rmse_log'] = r['test_rmse_log']
            row['test_mae_log'] = r['test_mae_log']
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{OUTPUT_DIR}/tpe_benchmark_summary.csv", index=False)

    # Print summary
    print(f"\n{'='*70}")
    print("TPE BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Dataset':<25} {'Task':<12} {'RMSE_orig':<12} {'RMSE_log':<12} {'MAE_log':<12} {'AUC':<8}")
    print("-"*85)

    for r in all_results:
        if r['task_type'] == 'classification':
            print(f"{r['dataset']:<25} {r['task_type']:<12} {'-':<12} {'-':<12} {'-':<12} {r['test_auc']:.4f}")
        else:
            print(f"{r['dataset']:<25} {r['task_type']:<12} {r['test_rmse_orig']:<12.4f} {r['test_rmse_log']:<12.4f} {r['test_mae_log']:<12.4f} {'-':<8}")

    print(f"\n{'='*70}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    run_tpe_benchmark()
