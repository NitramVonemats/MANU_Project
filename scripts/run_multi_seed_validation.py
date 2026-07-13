"""
Multi-Seed Validation Script
Run GNN training with multiple seeds for statistical validation

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score

warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKit_OK = True
except ImportError:
    RDKit_OK = False

# ============== CONFIGURATION ==============

DATASETS = {
    'Caco2_Wang': 'regression',
    'Half_Life_Obach': 'regression',
    'Clearance_Hepatocyte_AZ': 'regression',
    'Clearance_Microsome_AZ': 'regression',
    'tox21': 'classification',
    'herg': 'classification',
}

SEEDS = [42, 123, 456, 789, 1011]
MAX_EPOCHS = 50
PATIENCE = 12
OUTPUT_DIR = os.path.join(project_root, 'results', 'multi_seed')

# Best hyperparameters from HPO (extracted from runs/*.json best trials)
BEST_PARAMS = {
    'Caco2_Wang': {'hidden_dim': 96, 'num_layers': 5, 'lr': 5.685205e-03, 'weight_decay': 9.184901e-04, 'dropout': 0.0, 'head_dims': [512, 96, 96]},
    'Half_Life_Obach': {'hidden_dim': 256, 'num_layers': 4, 'lr': 1.833702e-03, 'weight_decay': 1.077335e-03, 'dropout': 0.0, 'head_dims': [384, 96, 64]},
    'Clearance_Hepatocyte_AZ': {'hidden_dim': 64, 'num_layers': 3, 'lr': 1.040259e-03, 'weight_decay': 4.268408e-03, 'dropout': 0.0, 'head_dims': [192, 128, 64]},
    'Clearance_Microsome_AZ': {'hidden_dim': 384, 'num_layers': 4, 'lr': 2.870875e-03, 'weight_decay': 1.216414e-03, 'dropout': 0.0, 'head_dims': [192, 128, 96]},
    'tox21': {'hidden_dim': 384, 'num_layers': 5, 'lr': 2.953687e-03, 'weight_decay': 1.713040e-03, 'dropout': 0.0, 'head_dims': [512, 192, 48]},
    'herg': {'hidden_dim': 512, 'num_layers': 5, 'lr': 8.938090e-03, 'weight_decay': 1.108049e-03, 'dropout': 0.0, 'head_dims': [384, 192, 48]},
}


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
            int(mw > 500), int(logp > 5), int(hbd > 5), int(hba > 10),
            Descriptors.MolMR(mol), Descriptors.BertzCT(mol),
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
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcTPSA(mol), Descriptors.NumRotatableBonds(mol),
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

    atoms = [atom_features(atom) for atom in mol.GetAtoms()]
    if len(atoms) == 0:
        return None
    x = torch.tensor(np.array(atoms), dtype=torch.float)

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([[i, j], [j, i]])

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    if dataset_name == "Caco2_Wang":
        descriptors = caco2_wang_descriptors(smiles)
    else:
        descriptors = adme_descriptors(smiles)

    adme_features = torch.tensor(descriptors, dtype=torch.float32).view(1, -1)

    return Data(x=x, edge_index=edge_index, adme_features=adme_features)


# ============== MODEL ==============

class GNNModel(nn.Module):
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

        graph_embed_dim = hidden_dim * 2
        combined_dim = graph_embed_dim + adme_dim

        # Dynamic MLP head (same as optimized_gnn.py)
        layers = []
        in_dim = combined_dim
        for h in head_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_mean = self.mean_pool(x, batch)
        x_max = self.max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        adme = data.adme_features.view(x.size(0), -1)
        x = torch.cat([x, adme], dim=1)

        return self.head(x)


# ============== DATA PREPARATION ==============

def is_classification_dataset(dataset_name: str) -> bool:
    classification_datasets = ['tox21', 'herg', 'clintox', 'ames', 'dili']
    return any(ds in dataset_name.lower() for ds in classification_datasets)


def prepare_data(dataset_name, val_fraction=0.1, seed=42, batch_size=32):
    """Prepare data with SAME preprocessing as original"""
    from tdc.single_pred import ADME, Tox
    from torch_geometric.loader import DataLoader

    is_classification = is_classification_dataset(dataset_name)

    if is_classification:
        tox_labels = {'tox21': 'NR-AR', 'herg': None}
        label = tox_labels.get(dataset_name.lower())
        if label:
            data_api = Tox(name=dataset_name, label_name=label)
        else:
            data_api = Tox(name=dataset_name)
    else:
        data_api = ADME(name=dataset_name)

    # TDC 2-way split (same as original)
    split = data_api.get_split(method="scaffold")

    # Create graphs
    train_graphs = []
    for _, row in split['train'].iterrows():
        g = smiles_to_graph(row['Drug'], dataset_name)
        if g is not None and not np.isnan(row['Y']):
            g.original_y = float(row['Y'])
            train_graphs.append(g)

    test_graphs = []
    for _, row in split['test'].iterrows():
        g = smiles_to_graph(row['Drug'], dataset_name)
        if g is not None and not np.isnan(row['Y']):
            g.original_y = float(row['Y'])
            test_graphs.append(g)

    # Manual train/val split with seed
    rng = random.Random(seed)
    rng.shuffle(train_graphs)

    n_val = max(1, int(len(train_graphs) * val_fraction))
    val_graphs = train_graphs[:n_val]
    train_graphs = train_graphs[n_val:]

    y_train = np.array([g.original_y for g in train_graphs], dtype=np.float32)
    y_val = np.array([g.original_y for g in val_graphs], dtype=np.float32)
    y_all = np.concatenate([y_train, y_val])

    if is_classification:
        for g in train_graphs + val_graphs + test_graphs:
            g.y = torch.tensor([float(g.original_y)], dtype=torch.float32)
        mu, sigma = 0.0, 1.0
        is_log_transformed = False
    else:
        all_negative = np.all(y_all < 0)

        if all_negative:
            y_log = y_train.astype(np.float32)
            mu = float(y_log.mean())
            sigma = float(y_log.std())
            if sigma < 1e-6:
                sigma = 1.0

            for g in train_graphs + val_graphs + test_graphs:
                g.y = torch.tensor([(g.original_y - mu) / sigma], dtype=torch.float32)
            is_log_transformed = True
        else:
            clip_min = 1e-3 if dataset_name != "Caco2_Wang" else 1e-6

            if dataset_name == "Caco2_Wang":
                positive_values = y_all[y_all > 0]
                if len(positive_values) > 0:
                    clip_min = max(float(positive_values.min()) / 1000.0, 1e-9)

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

    # ADME normalization
    adme_dim = train_graphs[0].adme_features.shape[1]
    adme_train = np.stack([g.adme_features.squeeze(0).numpy() for g in train_graphs])
    adme_mu = torch.tensor(adme_train.mean(0), dtype=torch.float32)
    adme_sigma = torch.tensor(adme_train.std(0), dtype=torch.float32)
    adme_sigma[adme_sigma == 0] = 1.0

    for g in train_graphs + val_graphs + test_graphs:
        g.adme_features = (g.adme_features - adme_mu.unsqueeze(0)) / adme_sigma.unsqueeze(0)

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


# ============== TRAINING & EVALUATION ==============

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


def evaluate(model, loader, device, is_classification, mu=0.0, sigma=1.0, is_log_transformed=False):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            preds.append(out.squeeze().cpu().numpy())
            labels.append(data.y.squeeze().cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    if is_classification:
        try:
            auc = roc_auc_score(labels, preds)
        except:
            auc = 0.5
        return {'auc': auc, 'val_metric': auc}
    else:
        preds_log = preds * sigma + mu
        labels_log = labels * sigma + mu

        preds_orig = np.exp(preds_log)
        labels_orig = np.exp(labels_log)

        rmse_log = np.sqrt(mean_squared_error(labels_log, preds_log))
        mae_log = mean_absolute_error(labels_log, preds_log)
        rmse_orig = np.sqrt(mean_squared_error(labels_orig, preds_orig))

        return {
            'rmse_log': rmse_log,
            'mae_log': mae_log,
            'rmse_orig': rmse_orig,
            'val_metric': rmse_orig,
        }


def run_single_seed(dataset_name, seed, params, device):
    """Run training for a single seed"""
    set_seed(seed)

    data_info = prepare_data(dataset_name, val_fraction=0.1, seed=seed, batch_size=32)
    is_classification = data_info['is_classification']
    mu, sigma = data_info['log_stats']

    model = GNNModel(
        input_dim=8,
        hidden_dim=params['hidden_dim'],
        output_dim=1,
        num_layers=params['num_layers'],
        dropout=params.get('dropout', 0.0),
        task_type='classification' if is_classification else 'regression',
        adme_dim=data_info['adme_dim'],
        head_dims=params.get('head_dims', (256, 128, 64)),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.BCEWithLogitsLoss() if is_classification else nn.MSELoss()

    best_val = float('inf') if not is_classification else 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        train_epoch(model, data_info['train_loader'], optimizer, criterion, device)
        val_metrics = evaluate(model, data_info['valid_loader'], device, is_classification, mu, sigma, data_info['is_log_transformed'])

        val_metric = val_metrics['auc'] if is_classification else val_metrics['val_metric']

        if (is_classification and val_metric > best_val) or (not is_classification and val_metric < best_val):
            best_val = val_metric
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, data_info['test_loader'], device, is_classification, mu, sigma, data_info['is_log_transformed'])

    return test_metrics


def run_multi_seed_validation(dataset_name, task_type):
    """Run multi-seed validation for a dataset"""
    print(f"\n{'='*60}")
    print(f"Multi-Seed Validation: {dataset_name} ({task_type})")
    print(f"Seeds: {SEEDS}")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = BEST_PARAMS.get(dataset_name, BEST_PARAMS['Caco2_Wang'])

    all_results = []

    for seed in SEEDS:
        print(f"\n  Running seed {seed}...")
        metrics = run_single_seed(dataset_name, seed, params, device)
        all_results.append(metrics)

        if task_type == 'classification':
            print(f"    AUC: {metrics['auc']:.4f}")
        else:
            print(f"    RMSE (log): {metrics['rmse_log']:.4f}, MAE (log): {metrics['mae_log']:.4f}")

    # Aggregate results
    if task_type == 'classification':
        aucs = [r['auc'] for r in all_results]
        summary = {
            'dataset': dataset_name,
            'task_type': task_type,
            'auc_mean': np.mean(aucs),
            'auc_std': np.std(aucs),
            'auc_values': aucs,
            'ci_lower': np.mean(aucs) - 1.96 * np.std(aucs) / np.sqrt(len(aucs)),
            'ci_upper': np.mean(aucs) + 1.96 * np.std(aucs) / np.sqrt(len(aucs)),
        }
    else:
        rmse_logs = [r['rmse_log'] for r in all_results]
        mae_logs = [r['mae_log'] for r in all_results]
        rmse_origs = [r['rmse_orig'] for r in all_results]

        summary = {
            'dataset': dataset_name,
            'task_type': task_type,
            'rmse_log_mean': np.mean(rmse_logs),
            'rmse_log_std': np.std(rmse_logs),
            'mae_log_mean': np.mean(mae_logs),
            'mae_log_std': np.std(mae_logs),
            'rmse_orig_mean': np.mean(rmse_origs),
            'rmse_orig_std': np.std(rmse_origs),
            'rmse_orig_median': float(np.median(rmse_origs)),
            'rmse_orig_values': [float(x) for x in rmse_origs],
            'rmse_log_values': rmse_logs,
            'mae_log_values': mae_logs,
            'ci_lower': np.mean(rmse_logs) - 1.96 * np.std(rmse_logs) / np.sqrt(len(rmse_logs)),
            'ci_upper': np.mean(rmse_logs) + 1.96 * np.std(rmse_logs) / np.sqrt(len(rmse_logs)),
        }

    print(f"\n  Summary:")
    if task_type == 'classification':
        print(f"    AUC: {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
        print(f"    95% CI: [{summary['ci_lower']:.4f}, {summary['ci_upper']:.4f}]")
    else:
        print(f"    RMSE (log): {summary['rmse_log_mean']:.4f} ± {summary['rmse_log_std']:.4f}")
        print(f"    MAE (log): {summary['mae_log_mean']:.4f} ± {summary['mae_log_std']:.4f}")
        print(f"    95% CI: [{summary['ci_lower']:.4f}, {summary['ci_upper']:.4f}]")

    return summary


def main():
    print("="*60)
    print("Multi-Seed Validation (FIXED - Consistent with Original HPO)")
    print("="*60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}

    for dataset_name, task_type in DATASETS.items():
        try:
            summary = run_multi_seed_validation(dataset_name, task_type)
            all_results[dataset_name] = summary
        except Exception as e:
            print(f"Error on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    with open(os.path.join(OUTPUT_DIR, 'multi_seed_results_fixed.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Create summary table
    rows = []
    for name, r in all_results.items():
        if r['task_type'] == 'classification':
            rows.append({
                'Dataset': name,
                'Task': 'classification',
                'AUC_Mean': r['auc_mean'],
                'AUC_Std': r['auc_std'],
                'CI_Lower': r['ci_lower'],
                'CI_Upper': r['ci_upper'],
            })
        else:
            rows.append({
                'Dataset': name,
                'Task': 'regression',
                'RMSE_log_Mean': r['rmse_log_mean'],
                'RMSE_log_Std': r['rmse_log_std'],
                'MAE_log_Mean': r['mae_log_mean'],
                'MAE_log_Std': r['mae_log_std'],
                'CI_Lower': r['ci_lower'],
                'CI_Upper': r['ci_upper'],
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, 'multi_seed_summary_fixed.csv'), index=False)

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
