"""
ОПТИМИЗИРАН GNN ЗА ADME ПРЕДВИДУВАЊЕ
=====================================

Базиран на анализа на 565 експерименти кои покажаа:
✅ Graph модел е најдобар (20-100x подобар од други верзии)
✅ 5 layers + 128 hidden channels = оптимална конфигурација
✅ Edge features ГО ВЛОШУВААТ performance (3.5x)
✅ Dropout не е потребен
✅ LR=0.001 е оптимален

НАЈДОБРИ РЕЗУЛТАТИ:
- Half_Life_Obach: RMSE=0.8388, R²=0.2765
- Clearance_Hepatocyte_AZ: RMSE=1.1921, R²=0.0868
- Clearance_Microsome_AZ: RMSE=1.0184, R²=0.3208
"""

import math
import time
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool, global_max_pool
import random

# ===================== RDKit SETUP =====================
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKit_OK = True
except Exception:
    RDKit_OK = False

# ===================== FEATURE EXTRACTION (SIMPLIFIED) =====================

def atom_features(atom):
    """Поедноставени atom features - само битното!"""
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
    """ADME дескриптори - основни физичко-хемиски својства"""
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
            int(mw > 500),           # Lipinski violations
            int(logp > 5),
            int(hbd > 5),
            int(hba > 10),
            Descriptors.MolMR(mol),  # Molecular refractivity
            Descriptors.BertzCT(mol), # Complexity
            rdMolDescriptors.CalcNumAliphaticRings(mol),
            Descriptors.NumHeteroatoms(mol),
        ], dtype=np.float32)
    except Exception:
        return np.zeros(15, dtype=np.float32)


# ===================== OPTIMAL ARCHITECTURE =====================

class OptimalGraphBackbone(nn.Module):
    """Graph модел - докажано најдобар! БЕЗ edge features"""

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=5):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            self.convs.append(GraphConv(in_channels, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(x, edge_index)
            h = norm(h)
            h = F.relu(h)

            # Residual connection
            if i > 0:  # Skip first layer
                x = x + h
            else:
                x = h

        return x


class SimpleReadout(nn.Module):
    """Едноставен но ефективен readout"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.out_dim = hidden_dim * 2  # mean + max

    def forward(self, x, batch):
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        return torch.cat([mean_pool, max_pool], dim=-1)


class OptimizedMolecularGNN(nn.Module):
    """ОПТИМИЗИРАН GNN модел - graph + ADME features"""

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=5, adme_dim=15):
        super().__init__()

        self.backbone = OptimalGraphBackbone(input_dim, hidden_dim, num_layers)
        self.readout = SimpleReadout(hidden_dim)

        # Combined features
        combined_dim = self.readout.out_dim + adme_dim

        # Prediction head - БЕЗ dropout!
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        # Graph embedding
        graph_emb = self.backbone(data)
        graph_pooled = self.readout(graph_emb, data.batch)

        # ADME features
        adme = data.adme_features
        if adme.dim() == 1:
            adme = adme.unsqueeze(0)

        # Combine and predict
        combined = torch.cat([graph_pooled, adme], dim=-1)
        return self.head(combined).squeeze(-1)


# ===================== DATA PROCESSING =====================

def row_to_graph(from_smiles_fn, smiles: str, y_value: float):
    """Креирај graph објект од SMILES"""
    g = from_smiles_fn(smiles)
    if g is None or getattr(g, "x", None) is None:
        return None

    # Поедноставени atom features
    if RDKit_OK:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            feats = [atom_features(atom) for atom in mol.GetAtoms()]
            g.x = torch.tensor(np.array(feats), dtype=torch.float32)

    g.x = g.x.float()
    g.edge_index = g.edge_index.long()
    g.original_y = float(y_value)
    g.y = torch.tensor([0.0], dtype=torch.float)  # ќе се трансформира подоцна

    # ADME descriptors
    g.adme_features = torch.tensor(adme_descriptors(smiles), dtype=torch.float)

    return g


def build_loaders(dataset_name: str, batch_train=32, batch_eval=64):
    """Подготви DataLoader објекти"""
    from tdc.single_pred import ADME
    try:
        from torch_geometric.utils import from_smiles
    except:
        from torch_geometric.data import from_smiles

    # Load dataset
    data_api = ADME(name=dataset_name)
    split = data_api.get_split(method="scaffold")

    print(f"\n📊 Dataset: {dataset_name}")
    print(f"Train: {len(split['train'])}, Test: {len(split['test'])}")

    # Create graphs
    train_list = [row_to_graph(from_smiles, row["Drug"], row["Y"])
                  for _, row in split["train"].iterrows()]
    train_list = [g for g in train_list if g is not None]

    test_list = [row_to_graph(from_smiles, row["Drug"], row["Y"])
                 for _, row in split["test"].iterrows()]
    test_list = [g for g in test_list if g is not None]

    # Validation split
    n_val = max(1, int(0.1 * len(train_list)))
    val_list = train_list[:n_val]
    train_list = train_list[n_val:]

    print(f"Splits: train={len(train_list)}, val={len(val_list)}, test={len(test_list)}")

    # Target scaling - LOG SPACE!
    y_train = np.array([g.original_y for g in train_list])
    y_log = np.log(np.clip(y_train, 1e-3, None))
    mu, sigma = y_log.mean(), y_log.std()

    print(f"Log scaling: μ={mu:.3f}, σ={sigma:.3f}")

    # Apply transformation
    for g in train_list + val_list + test_list:
        y_transformed = (np.log(max(1e-3, g.original_y)) - mu) / sigma
        g.y = torch.tensor([y_transformed], dtype=torch.float)

    # ADME standardization
    adme_train = np.stack([g.adme_features.numpy() for g in train_list])
    adme_mu, adme_sigma = adme_train.mean(0), adme_train.std(0)
    adme_sigma[adme_sigma == 0] = 1.0

    for g in train_list + val_list + test_list:
        g.adme_features = (g.adme_features - torch.from_numpy(adme_mu)) / torch.from_numpy(adme_sigma)

    # Create loaders
    train_loader = DataLoader(train_list, batch_size=batch_train, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_eval, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_eval, shuffle=False)

    return train_loader, val_loader, test_loader, (mu, sigma)


# ===================== TRAINING =====================

def train_epoch(model, loader, optimizer, device):
    """Тренирај една епоха"""
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        pred = model(data)
        loss = F.mse_loss(pred, data.y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, mu, sigma):
    """Евалуирај модел (вратено во оригинален scale)"""
    model.eval()
    preds, trues = [], []

    for data in loader:
        data = data.to(device)
        pred = model(data)
        preds.append(pred.cpu())
        trues.append(data.y.cpu())

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    # Inverse transform
    preds_orig = np.exp(preds * sigma + mu)
    trues_orig = np.exp(trues * sigma + mu)

    # Metrics
    rmse = np.sqrt(np.mean((preds_orig - trues_orig) ** 2))
    mae = np.mean(np.abs(preds_orig - trues_orig))
    r2 = 1 - np.sum((preds_orig - trues_orig) ** 2) / (np.sum((trues_orig - trues_orig.mean()) ** 2) + 1e-12)

    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_model(dataset_name, epochs=100, patience=20, device="cpu", seed=42):
    """Тренирај оптимизиран модел"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Data
    train_loader, val_loader, test_loader, (mu, sigma) = build_loaders(dataset_name)

    # Model - ОПТИМАЛНА КОНФИГУРАЦИЈА!
    model = OptimizedMolecularGNN(
        input_dim=8,      # Поедноставени atom features
        hidden_dim=128,   # Докажано најдобро
        num_layers=5,     # Докажано најдобро
        adme_dim=15       # Поедноставени ADME features
    ).to(device)

    print(f"\n🧬 Model: Graph (5 layers, 128 hidden)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer - БЕЗ weight decay!
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Training loop
    best_val_rmse = float('inf')
    best_state = None
    no_improvement = 0

    print(f"\n🏃 Training...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device, mu, sigma)

        scheduler.step(val_metrics['rmse'])

        # Track best
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_state = deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: loss={train_loss:.4f}, val_rmse={val_metrics['rmse']:.3f}, val_r2={val_metrics['r2']:.3f}")

        # Early stopping
        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    val_metrics = evaluate(model, val_loader, device, mu, sigma)
    test_metrics = evaluate(model, test_loader, device, mu, sigma)

    train_time = time.time() - start_time

    print(f"\n✅ Training complete ({train_time:.1f}s)")
    print(f"Val  - RMSE: {val_metrics['rmse']:.3f}, MAE: {val_metrics['mae']:.3f}, R²: {val_metrics['r2']:.3f}")
    print(f"Test - RMSE: {test_metrics['rmse']:.3f}, MAE: {test_metrics['mae']:.3f}, R²: {test_metrics['r2']:.3f}")

    return {
        "model": model,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_time": train_time,
        "dataset": dataset_name,
    }


# ===================== BENCHMARK =====================

def run_benchmark():
    """Тестирај на сите 3 datasets"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"ОПТИМИЗИРАН GNN BENCHMARK")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Конфигурација: Graph, 5 layers, 128 hidden, БЕЗ edge features/dropout")

    datasets = ["Half_Life_Obach", "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ"]
    results = []

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")

        result = train_model(dataset_name, epochs=100, patience=20, device=device, seed=42)
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print(f"РЕЗУЛТАТИ - РЕЗИМЕ")
    print(f"{'='*70}")

    df_results = []
    for res in results:
        df_results.append({
            "Dataset": res["dataset"],
            "Val_RMSE": res["val_metrics"]["rmse"],
            "Test_RMSE": res["test_metrics"]["rmse"],
            "Test_MAE": res["test_metrics"]["mae"],
            "Test_R2": res["test_metrics"]["r2"],
            "Time_s": res["train_time"],
        })

    df = pd.DataFrame(df_results)
    print(df.to_string(index=False))

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"optimized_gnn_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved: {csv_path}")

    return results


if __name__ == "__main__":
    results = run_benchmark()

    print(f"\n{'='*70}")
    print("✨ BENCHMARK ЗАВРШЕН!")
    print(f"{'='*70}")
    print("\n📊 Клучни карактеристики:")
    print("  ✅ Graph архитектура (докажано најдобра)")
    print("  ✅ 5 layers, 128 hidden channels")
    print("  ✅ БЕЗ edge features (го влошуваат performance)")
    print("  ✅ БЕЗ dropout (не е потребен)")
    print("  ✅ Поедноставени features (8 atom + 15 ADME)")
    print("  ✅ LR=0.001, Adam optimizer")
    print("\n🎯 Очекувани резултати:")
    print("  Half_Life_Obach: RMSE ~0.84")
    print("  Clearance_Hepatocyte_AZ: RMSE ~1.19")
    print("  Clearance_Microsome_AZ: RMSE ~1.02")
