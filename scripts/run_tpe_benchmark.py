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
from sklearn.metrics import mean_squared_error, roc_auc_score
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


def prepare_data(split, batch_size=32):
    """Prepare PyG DataLoaders from TDC split"""
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


def evaluate(model, loader, device, task_type):
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
        return roc_auc_score(labels, preds)
    else:
        return np.sqrt(mean_squared_error(labels, preds))


class TPEObjective:
    """Objective function for TPE optimization"""

    def __init__(self, dataset_name, task_type, device):
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.device = device

        # Load data once
        print(f"  Loading {dataset_name}...")
        split = get_dataset(dataset_name)
        self.train_loader, self.valid_loader, self.test_loader = prepare_data(split)
        print(f"  Train: {len(self.train_loader.dataset)}, Valid: {len(self.valid_loader.dataset)}, Test: {len(self.test_loader.dataset)}")

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
            val_metric = evaluate(model, self.valid_loader, self.device, self.task_type)

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

    test_metric = evaluate(model, objective.test_loader, device, task_type)
    print(f"  Test metric: {test_metric:.4f}")

    return {
        'dataset': dataset_name,
        'algorithm': 'TPE',
        'task_type': task_type,
        'best_params': best_params,
        'best_val_metric': study.best_value,
        'test_metric': test_metric,
        'n_trials': len(study.trials),
        'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'optimization_history': [
            {'trial': t.number, 'value': t.value, 'params': t.params}
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
    }


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

    # Save summary
    summary_df = pd.DataFrame([{
        'dataset': r['dataset'],
        'algorithm': 'TPE',
        'task_type': r['task_type'],
        'test_metric': r['test_metric'],
        'best_val_metric': r['best_val_metric'],
        'hidden_dim': r['best_params']['hidden_dim'],
        'num_layers': r['best_params']['num_layers'],
        'lr': r['best_params']['lr'],
        'dropout': r['best_params']['dropout'],
        'n_trials': r['n_trials'],
        'n_pruned': r['n_pruned']
    } for r in all_results])

    summary_df.to_csv(f"{OUTPUT_DIR}/tpe_benchmark_summary.csv", index=False)

    # Print summary
    print(f"\n{'='*70}")
    print("TPE BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Dataset':<30} {'Task':<15} {'Test Metric':<15} {'Best Params'}")
    print("-"*80)

    for r in all_results:
        metric_name = 'AUC' if r['task_type'] == 'classification' else 'RMSE'
        params = f"h={r['best_params']['hidden_dim']}, L={r['best_params']['num_layers']}"
        print(f"{r['dataset']:<30} {r['task_type']:<15} {r['test_metric']:.4f} ({metric_name}){'':<5} {params}")

    print(f"\n{'='*70}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    run_tpe_benchmark()
