"""
TPE (Tree-structured Parzen Estimator) Hyperparameter Optimization
Using Optuna for Bayesian optimization on MANU benchmark

FIXED VERSION: Uses same preprocessing as original HPO (optimized_gnn.py)
- Same atom features (8 features)
- Same ADME descriptors
- Same data splitting (TDC 2-way, then manual train/val split)
- Same log transformation and clipping
- Same model architecture (mean+max pool, ADME features)
"""

import os
import sys
import json
import time
import random
import warnings
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKit_OK = True
except ImportError:
    RDKit_OK = False
    print("[WARNING] RDKit not available")

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============== FEATURE EXTRACTION (SAME AS ORIGINAL) ==============

def atom_features(atom):
    """Same 8 atom features as original optimized_gnn.py"""
    try:
        return np.array([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            atom.GetTotalNumHs(),
            atom.GetMass(),
        ], dtype=np.float32)
    except Exception:
        return np.zeros(8, dtype=np.float32)


def adme_descriptors(smiles: str) -> np.ndarray:
    """Same 15 ADME descriptors as original"""
    if not RDKit_OK:
        return np.zeros(15, dtype=np.float32)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(15, dtype=np.float32)

    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        rotatable = Descriptors.NumRotatableBonds(mol)
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

        return np.array([
            mw, logp, hbd, hba, tpsa, rotatable, aromatic_rings,
            int(mw > 500),
            int(logp > 5),
            int(hbd > 5),
            int(hba > 10),
            Descriptors.MolMR(mol),
            Descriptors.BertzCT(mol),
            rdMolDescriptors.CalcNumAliphaticRings(mol),
            Descriptors.NumHeteroatoms(mol),
        ], dtype=np.float32)
    except Exception:
        return np.zeros(15, dtype=np.float32)


def caco2_wang_descriptors(smiles: str) -> np.ndarray:
    """Same 7 Caco2 descriptors as original"""
    if not RDKit_OK:
        return np.zeros(7, dtype=np.float32)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(7, dtype=np.float32)

    try:
        return np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
        ], dtype=np.float32)
    except Exception:
        return np.zeros(7, dtype=np.float32)


def smiles_to_graph(smiles, dataset_name=None):
    """Convert SMILES to PyG Data object with ADME features"""
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features (8 features, same as original)
    atoms = [atom_features(atom) for atom in mol.GetAtoms()]
    if len(atoms) == 0:
        return None
    x = torch.tensor(np.array(atoms), dtype=torch.float)

    # Edges
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([[i, j], [j, i]])

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # ADME descriptors (same as original)
    if dataset_name == "Caco2_Wang":
        descriptors = caco2_wang_descriptors(smiles)
    else:
        descriptors = adme_descriptors(smiles)

    adme_features = torch.tensor(descriptors, dtype=torch.float32).view(1, -1)

    return Data(x=x, edge_index=edge_index, adme_features=adme_features)


# ============== MODEL (SAME ARCHITECTURE AS ORIGINAL) ==============

class GNNModel(nn.Module):
    """GCN model matching original optimized_gnn.py architecture"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout,
                 task_type, adme_dim=15, head_dims=(256, 128, 64)):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout
        self.task_type = task_type
        self.mean_pool = global_mean_pool
        self.max_pool = global_max_pool

        # Readout: mean + max pooling (same as original)
        graph_embed_dim = hidden_dim * 2  # mean + max
        combined_dim = graph_embed_dim + adme_dim

        # Head with configurable dimensions
        layers = []
        prev_dim = combined_dim
        for dim in head_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Mean + Max pooling (same as original)
        x_mean = self.mean_pool(x, batch)
        x_max = self.max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Concatenate ADME features
        adme = data.adme_features.view(x.size(0), -1)
        x = torch.cat([x, adme], dim=1)

        return self.head(x)


# ============== DATA PREPARATION (SAME AS ORIGINAL) ==============

def is_classification_dataset(dataset_name: str) -> bool:
    classification_datasets = ['tox21', 'herg', 'clintox', 'ames', 'dili']
    return any(ds in dataset_name.lower() for ds in classification_datasets)


def prepare_data(dataset_name, val_fraction=0.1, seed=42, batch_size=32):
    """Prepare data with SAME preprocessing as original optimized_gnn.py

    Key differences from broken version:
    1. Uses TDC 2-way split (train/test), then manually splits train into train/val
    2. Uses 8 atom features (not 9)
    3. Includes ADME descriptors
    4. Uses clip_min=1e-3 for non-Caco2
    5. Uses y_all (train+val) for log detection
    """
    from tdc.single_pred import ADME, Tox
    from torch_geometric.loader import DataLoader

    is_classification = is_classification_dataset(dataset_name)

    # Load TDC data (2-way split like original)
    if is_classification:
        tox_labels = {'tox21': 'NR-AR', 'herg': None}
        label = tox_labels.get(dataset_name.lower())
        if label:
            data_api = Tox(name=dataset_name, label_name=label)
        else:
            data_api = Tox(name=dataset_name)
    else:
        data_api = ADME(name=dataset_name)

    split = data_api.get_split(method="scaffold")  # Returns train/test only

    print(f"\n[DATA] Dataset: {dataset_name}")
    print(f"  TDC Split: Train={len(split['train'])}, Test={len(split['test'])}")

    # Create graphs
    train_graphs = []
    for _, row in split['train'].iterrows():
        g = smiles_to_graph(row['Drug'], dataset_name)
        if g is not None:
            y = row['Y']
            if not np.isnan(y):
                g.original_y = float(y)
                train_graphs.append(g)

    test_graphs = []
    for _, row in split['test'].iterrows():
        g = smiles_to_graph(row['Drug'], dataset_name)
        if g is not None:
            y = row['Y']
            if not np.isnan(y):
                g.original_y = float(y)
                test_graphs.append(g)

    # Manual train/val split (SAME as original)
    rng = random.Random(seed)
    rng.shuffle(train_graphs)

    n_val = max(1, int(len(train_graphs) * val_fraction))
    if len(train_graphs) - n_val < 1:
        n_val = max(1, len(train_graphs) - 1)
    val_graphs = train_graphs[:n_val]
    train_graphs = train_graphs[n_val:]

    print(f"  After split: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")

    # Get y values
    y_train = np.array([g.original_y for g in train_graphs], dtype=np.float32)
    y_val = np.array([g.original_y for g in val_graphs], dtype=np.float32)
    y_all = np.concatenate([y_train, y_val])  # Use train+val for detection (same as original)

    # Classification: no transformation
    if is_classification:
        for g in train_graphs + val_graphs + test_graphs:
            g.y = torch.tensor([float(g.original_y)], dtype=torch.float32)
        mu, sigma = 0.0, 1.0
        is_log_transformed = False
    else:
        # Regression: detect if already log-transformed
        all_negative = np.all(y_all < 0)

        if all_negative:
            # Already log-transformed (e.g., Caco2_Wang)
            print(f"  [DATA] Detected log-transformed values (all negative)")
            y_log = y_train.astype(np.float32)
            mu = float(y_log.mean())
            sigma = float(y_log.std())
            if sigma < 1e-6:
                sigma = 1.0

            for g in train_graphs + val_graphs + test_graphs:
                g.y = torch.tensor([(g.original_y - mu) / sigma], dtype=torch.float32)
            is_log_transformed = True
        else:
            # Positive values: apply log transform
            # Use clip_min=1e-3 for non-Caco2 (SAME as original)
            clip_min = 1e-3 if dataset_name != "Caco2_Wang" else 1e-6

            if dataset_name == "Caco2_Wang":
                positive_values = y_all[y_all > 0]
                if len(positive_values) > 0:
                    min_val = float(positive_values.min())
                    clip_min = max(min_val / 1000.0, 1e-9)
                print(f"  Using clip_min={clip_min:.10e}")

            y_train_clipped = np.clip(y_train, clip_min, None)
            y_log = np.log(y_train_clipped)
            mu = float(y_log.mean())
            sigma = float(y_log.std())
            if sigma < 1e-6:
                sigma = 1.0

            for g in train_graphs + val_graphs + test_graphs:
                y_value = max(clip_min, float(g.original_y))
                g.y = torch.tensor([(np.log(y_value) - mu) / sigma], dtype=torch.float32)
            is_log_transformed = False

    # ADME feature normalization (SAME as original)
    adme_dim = train_graphs[0].adme_features.shape[1]
    adme_train = np.stack([g.adme_features.squeeze(0).numpy() for g in train_graphs])
    adme_mu = torch.tensor(adme_train.mean(0), dtype=torch.float32)
    adme_sigma = torch.tensor(adme_train.std(0), dtype=torch.float32)
    adme_sigma[adme_sigma == 0] = 1.0

    for g in train_graphs + val_graphs + test_graphs:
        g.adme_features = (g.adme_features - adme_mu.unsqueeze(0)) / adme_sigma.unsqueeze(0)

    print(f"  Log scaling: mu={mu:.3f}, sigma={sigma:.3f}")
    print(f"  ADME features: {adme_dim} dimensions")

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_graphs, batch_size=batch_size)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)

    return {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'test_loader': test_loader,
        'log_stats': (mu, sigma),
        'is_log_transformed': is_log_transformed,
        'adme_dim': adme_dim,
        'is_classification': is_classification,
    }


# ============== TRAINING ==============

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out.squeeze(), data.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, is_classification, mu=0.0, sigma=1.0,
             is_log_transformed=False):
    """Evaluate with proper inverse transform (SAME as original)"""
    model.eval()
    preds, labels, originals = [], [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            preds.append(out.squeeze().cpu().numpy())
            labels.append(data.y.squeeze().cpu().numpy())
            if hasattr(data, 'original_y'):
                originals.extend([g.original_y for g in data.to_data_list()])

    preds = np.concatenate(preds) if len(preds) > 0 else np.array([])
    labels = np.concatenate(labels) if len(labels) > 0 else np.array([])

    if is_classification:
        try:
            auc = roc_auc_score(labels, preds)
        except:
            auc = 0.5
        return {'auc': auc, 'val_metric': auc}
    else:
        # Denormalize to log scale
        preds_log = preds * sigma + mu
        labels_log = labels * sigma + mu

        # Convert to original scale
        if is_log_transformed:
            # Values were already in log space, just exp
            preds_orig = np.exp(preds_log)
            labels_orig = np.exp(labels_log)
        else:
            # We applied log, so exp to get original
            preds_orig = np.exp(preds_log)
            labels_orig = np.exp(labels_log)

        # Metrics on LOG scale (TDC standard)
        rmse_log = np.sqrt(mean_squared_error(labels_log, preds_log))
        mae_log = mean_absolute_error(labels_log, preds_log)

        # Metrics on ORIGINAL scale
        rmse_orig = np.sqrt(mean_squared_error(labels_orig, preds_orig))
        mae_orig = mean_absolute_error(labels_orig, preds_orig)

        return {
            'rmse_log': rmse_log,
            'mae_log': mae_log,
            'rmse_orig': rmse_orig,
            'mae_orig': mae_orig,
            'val_metric': rmse_orig,  # Use original scale for early stopping (same as original)
        }


# ============== TPE OPTIMIZATION ==============

def objective(trial, data_info, device):
    """Optuna objective function"""
    # Hyperparameter search space (same as original HPO)
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 384, 512])
    num_layers = trial.suggest_int('num_layers', 2, 8)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)

    # Model
    is_classification = data_info['is_classification']
    adme_dim = data_info['adme_dim']
    mu, sigma = data_info['log_stats']

    model = GNNModel(
        input_dim=8,  # 8 atom features (same as original)
        hidden_dim=hidden_dim,
        output_dim=1,
        num_layers=num_layers,
        dropout=dropout,
        task_type='classification' if is_classification else 'regression',
        adme_dim=adme_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if is_classification:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    # Training
    best_val = float('inf') if not is_classification else 0.0
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        train_epoch(model, data_info['train_loader'], optimizer, criterion, device)

        val_metrics = evaluate(
            model, data_info['valid_loader'], device, is_classification,
            mu, sigma, data_info['is_log_transformed']
        )

        if is_classification:
            val_metric = val_metrics['auc']
            improved = val_metric > best_val
        else:
            val_metric = val_metrics['val_metric']  # RMSE on original scale
            improved = val_metric < best_val

        if improved:
            best_val = val_metric
            patience_counter = 0
        else:
            patience_counter += 1

        # Pruning
        trial.report(val_metric if is_classification else -val_metric, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if patience_counter >= PATIENCE:
            break

    return best_val if is_classification else -best_val  # Optuna maximizes


def run_tpe_benchmark(dataset_name, task_type):
    """Run TPE optimization for a single dataset"""
    print(f"\n{'='*60}")
    print(f"TPE Benchmark: {dataset_name} ({task_type})")
    print(f"{'='*60}")

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Prepare data
    data_info = prepare_data(dataset_name, val_fraction=0.1, seed=SEED, batch_size=32)

    # Create study
    sampler = TPESampler(seed=SEED, multivariate=True)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    study = optuna.create_study(
        direction='maximize' if task_type == 'classification' else 'minimize',
        sampler=sampler,
        pruner=pruner,
    )

    # Optimize
    start_time = time.time()
    study.optimize(
        lambda trial: objective(trial, data_info, device),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    total_time = time.time() - start_time

    # Best trial
    best_trial = study.best_trial
    print(f"\nBest trial: {best_trial.number}")
    print(f"Best value: {best_trial.value:.4f}")
    print(f"Best params: {best_trial.params}")

    # Retrain with best params and evaluate on test
    set_seed(SEED)
    is_classification = data_info['is_classification']
    mu, sigma = data_info['log_stats']

    best_model = GNNModel(
        input_dim=8,
        hidden_dim=best_trial.params['hidden_dim'],
        output_dim=1,
        num_layers=best_trial.params['num_layers'],
        dropout=best_trial.params['dropout'],
        task_type='classification' if is_classification else 'regression',
        adme_dim=data_info['adme_dim'],
    ).to(device)

    optimizer = torch.optim.Adam(
        best_model.parameters(),
        lr=best_trial.params['lr'],
        weight_decay=best_trial.params['weight_decay'],
    )
    criterion = nn.BCEWithLogitsLoss() if is_classification else nn.MSELoss()

    # Train
    best_val = float('inf') if not is_classification else 0.0
    for epoch in range(MAX_EPOCHS):
        train_epoch(best_model, data_info['train_loader'], optimizer, criterion, device)
        val_metrics = evaluate(
            best_model, data_info['valid_loader'], device, is_classification,
            mu, sigma, data_info['is_log_transformed']
        )
        val_metric = val_metrics['auc'] if is_classification else val_metrics['val_metric']
        if (is_classification and val_metric > best_val) or (not is_classification and val_metric < best_val):
            best_val = val_metric
            best_state = best_model.state_dict().copy()

    best_model.load_state_dict(best_state)

    # Test evaluation
    test_metrics = evaluate(
        best_model, data_info['test_loader'], device, is_classification,
        mu, sigma, data_info['is_log_transformed']
    )

    print(f"\nTest Results:")
    if is_classification:
        print(f"  AUC: {test_metrics['auc']:.4f}")
    else:
        print(f"  RMSE (log): {test_metrics['rmse_log']:.4f}")
        print(f"  MAE (log): {test_metrics['mae_log']:.4f}")
        print(f"  RMSE (orig): {test_metrics['rmse_orig']:.4f}")

    # Save results
    results = {
        'dataset': dataset_name,
        'task_type': task_type,
        'best_trial': best_trial.number,
        'best_params': best_trial.params,
        'test_metrics': test_metrics,
        'n_trials': N_TRIALS,
        'total_time': total_time,
        'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f'tpe_{dataset_name}_results_fixed.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main():
    print("="*60)
    print("TPE Benchmark (FIXED - Consistent with Original HPO)")
    print("="*60)

    all_results = []

    for dataset_name, task_type in DATASETS.items():
        try:
            results = run_tpe_benchmark(dataset_name, task_type)
            all_results.append(results)
        except Exception as e:
            print(f"Error on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    summary_data = []
    for r in all_results:
        if r['task_type'] == 'classification':
            summary_data.append({
                'Dataset': r['dataset'],
                'Task': r['task_type'],
                'AUC': r['test_metrics'].get('auc', 0),
            })
        else:
            summary_data.append({
                'Dataset': r['dataset'],
                'Task': r['task_type'],
                'RMSE_log': r['test_metrics'].get('rmse_log', 0),
                'MAE_log': r['test_metrics'].get('mae_log', 0),
                'RMSE_orig': r['test_metrics'].get('rmse_orig', 0),
            })

    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))

    # Save summary
    df.to_csv(os.path.join(OUTPUT_DIR, 'tpe_benchmark_summary_fixed.csv'), index=False)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
