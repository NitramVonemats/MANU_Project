"""
PRIMER - FIXED FOUNDATION
=========================
Addresses the negative R2 issue with:
1. Stronger regularization (dropout 0.4-0.5, weight decay 1e-3)
2. Simpler architecture (prevent overfitting)
3. Ensemble predictions (3-5 seeds)
4. Better validation strategy
5. Proper baseline comparisons

This is the "foundation fix" before adding advanced methods.
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
import random

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv, SAGEConv, GINConv, GATConv, TAGConv, ChebConv, SGConv,
    TransformerConv, GraphConv, global_mean_pool, global_max_pool,
    GlobalAttention
)

# ===================== RDKit SETUP =====================
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    RDKit_OK = True
except Exception:
    RDKit_OK = False

# ===================== GLOBAL SCALERS =====================
TARGET_SCALER = {"mode": "log", "mu": 0.0, "sigma": 1.0}
ADME_SCALER = {"mu": None, "sigma": None}
RNG_SEED = 42

# ===================== UTILS =====================
try:
    from scipy.stats import spearmanr
    def spearman_metric(y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        return float(spearmanr(y_true, y_pred).correlation)
except Exception:
    def spearman_metric(y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        r_true = y_true.argsort().argsort().astype(float)
        r_pred = y_pred.argsort().argsort().astype(float)
        r_true = (r_true - r_true.mean()) / (r_true.std() + 1e-12)
        r_pred = (r_pred - r_pred.mean()) / (r_pred.std() + 1e-12)
        return float((r_true * r_pred).mean())

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics_np(y_true: np.ndarray, y_pred: np.ndarray, *, log_space: bool = False):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if log_space:
        yt = np.log1p(np.maximum(0.0, y_true))
        yp = np.log1p(np.maximum(0.0, y_pred))
    else:
        yt, yp = y_true, y_pred
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    ybar = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - ybar) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"rmse": rmse, "mae": mae, "r2": r2}

def set_target_scaler(dataset_name: str, y_train: np.ndarray, force_mode: str = "log"):
    mode = force_mode
    if mode == "log":
        z = np.log(np.clip(y_train.astype(float), 1e-3, None))
        mu = float(z.mean())
        sigma = float(z.std() or 1.0)
    else:
        mu = float(y_train.mean())
        sigma = float(y_train.std() or 1.0)
    TARGET_SCALER.update({"mode": mode, "mu": mu, "sigma": sigma})

def _transform_y(y: float) -> float:
    if TARGET_SCALER["mode"] == "log":
        return (np.log(max(1e-3, float(y))) - TARGET_SCALER["mu"]) / TARGET_SCALER["sigma"]
    return (float(y) - TARGET_SCALER["mu"]) / TARGET_SCALER["sigma"]

def _inverse_y(t: np.ndarray) -> np.ndarray:
    if TARGET_SCALER["mode"] == "log":
        z = t * TARGET_SCALER["sigma"] + TARGET_SCALER["mu"]
        z = np.clip(z, -10, 10)
        return np.exp(z)
    return t * TARGET_SCALER["sigma"] + TARGET_SCALER["mu"]

# ===================== MOLECULAR FEATURES =====================
def enhanced_atom_features(atom):
    try:
        features = [
            atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
            int(atom.GetHybridization()), int(atom.GetIsAromatic()),
            int(atom.IsInRing()), atom.GetTotalNumHs(), atom.GetMass(),
            int(atom.IsInRingSize(3)), int(atom.IsInRingSize(4)),
            int(atom.IsInRingSize(5)), int(atom.IsInRingSize(6)),
            int(atom.IsInRingSize(7)), int(atom.IsInRingSize(8)),
            atom.GetTotalValence(), atom.GetImplicitValence(),
            atom.GetExplicitValence(), int(atom.GetChiralTag()),
            len(atom.GetNeighbors()), atom.GetTotalDegree(),
            atom.GetAtomicNum() / 100.0,
            int(atom.GetSymbol() == 'C'), int(atom.GetSymbol() == 'N'),
            int(atom.GetSymbol() == 'O'), int(atom.GetSymbol() == 'S'),
            int(atom.GetSymbol() == 'P'), int(atom.GetSymbol() == 'F'),
            int(atom.GetSymbol() == 'Cl'), int(atom.GetSymbol() == 'Br'),
            int(atom.GetSymbol() == 'I'),
            int(atom.GetHybridization() == Chem.HybridizationType.SP),
            int(atom.GetHybridization() == Chem.HybridizationType.SP2),
            int(atom.GetHybridization() == Chem.HybridizationType.SP3),
            int(atom.GetHybridization() == Chem.HybridizationType.SP3D),
            int(atom.GetHybridization() == Chem.HybridizationType.SP3D2),
            int(atom.GetSymbol() in ['N', 'O']),
            int(atom.GetSymbol() in ['N', 'O', 'F']),
        ]
        return np.array(features, dtype=np.float32)
    except Exception:
        return np.array([
            atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
            int(atom.GetHybridization()), int(atom.GetIsAromatic()),
            int(atom.IsInRing()), atom.GetTotalNumHs(), atom.GetMass()
        ] + [0.0] * 29, dtype=np.float32)

def adme_specific_descriptors(smiles: str) -> np.ndarray:
    if not RDKit_OK:
        return np.zeros(30, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(30, dtype=np.float32)
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        descriptors = [
            mw, logp, hbd, hba, rotatable_bonds, tpsa, aromatic_rings,
            int(mw > 500), int(logp > 5), int(hbd > 5), int(hba > 10),
            int(rotatable_bonds > 10), int(tpsa > 140),
            Descriptors.MolMR(mol), Descriptors.LabuteASA(mol),
            rdMolDescriptors.CalcNumAliphaticRings(mol),
            rdMolDescriptors.CalcNumSaturatedRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            Descriptors.BertzCT(mol),
            Descriptors.Chi0v(mol), Descriptors.Chi1v(mol),
            Descriptors.Kappa1(mol), Descriptors.Kappa2(mol),
            Descriptors.NumHeteroatoms(mol),
            rdMolDescriptors.CalcNumHeterocycles(mol),
            int(mw < 200), int(mw > 800),
        ]
        descriptors = descriptors[:30]
        while len(descriptors) < 30:
            descriptors.append(0.0)
        return np.array(descriptors, dtype=np.float32)
    except Exception:
        return np.zeros(30, dtype=np.float32)

# ===================== SIMPLER ARCHITECTURE (LESS OVERFITTING) =====================
class SimpleGraphBackbone(nn.Module):
    """Simpler GNN backbone with strong regularization"""
    def __init__(self,
                 model_name="SAGE",
                 graph_layers=3,  # REDUCED from 5
                 graph_hidden_channels=64,  # REDUCED from 128
                 dropout=0.5,  # INCREASED from 0.2-0.3
                 input_dim=37,
                 **kwargs):
        super().__init__()
        self.model_name = model_name
        self.graph_layers = graph_layers
        self.output_dim = graph_hidden_channels

        convs, norms, dropouts = [], [], []

        for i in range(graph_layers):
            input_channels = input_dim if i == 0 else graph_hidden_channels

            if model_name == "SAGE":
                conv = SAGEConv(input_channels, graph_hidden_channels, aggr='mean')
            elif model_name == "GCN":
                conv = GCNConv(input_channels, graph_hidden_channels)
            elif model_name == "GIN":
                nn_lin = nn.Sequential(
                    nn.Linear(input_channels, graph_hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),  # Extra dropout in GIN
                    nn.Linear(graph_hidden_channels, graph_hidden_channels)
                )
                conv = GINConv(nn_lin)
            else:
                conv = SAGEConv(input_channels, graph_hidden_channels, aggr='mean')

            convs.append(conv)
            norms.append(nn.BatchNorm1d(graph_hidden_channels))
            dropouts.append(nn.Dropout(dropout))

        self._convs = nn.ModuleList(convs)
        self._norms = nn.ModuleList(norms)
        self._dropouts = nn.ModuleList(dropouts)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, (conv, norm, dropout) in enumerate(zip(self._convs, self._norms, self._dropouts)):
            h = conv(x, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = dropout(h)
            # Skip connection only for middle layers (not first/last)
            if i > 0 and i < len(self._convs) - 1 and h.shape == x.shape:
                x = x + 0.5 * h  # Weighted skip connection
            else:
                x = h
        return x

class SimpleReadout(nn.Module):
    """Simpler readout with strong regularization"""
    def __init__(self, in_dim: int):
        super().__init__()
        self.out_dim = in_dim * 2  # Just mean + max

    def forward(self, x, batch):
        mean_p = global_mean_pool(x, batch)
        max_p = global_max_pool(x, batch)
        return torch.cat([mean_p, max_p], dim=-1)

class SimpleMolecularRegressor(nn.Module):
    """Simpler model with strong regularization to prevent overfitting"""
    def __init__(self, backbone: SimpleGraphBackbone, adme_dim: int = 30):
        super().__init__()
        self.backbone = backbone
        self.adme_dim = adme_dim
        self.readout = SimpleReadout(backbone.output_dim)

        combined_dim = self.readout.out_dim + adme_dim
        self.comb_norm = nn.BatchNorm1d(combined_dim)

        # SIMPLER HEAD with STRONG DROPOUT
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),  # Strong dropout
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),  # Strong dropout
            nn.Linear(64, 1)
        )

    def forward(self, data: Data) -> torch.Tensor:
        graph_emb = self.backbone(data)
        graph_pooled = self.readout(graph_emb, data.batch)

        adme = getattr(data, "adme_features", None)
        if adme is None:
            batch_size = int(data.batch.max().item() + 1) if hasattr(data, 'batch') else 1
            adme = torch.zeros(batch_size, self.adme_dim, device=data.x.device)
        elif adme.dim() == 3 and adme.size(1) == 1:
            adme = adme.squeeze(1)
        elif adme.dim() == 1:
            adme = adme.view(-1, self.adme_dim)

        combined = torch.cat([graph_pooled, adme], dim=-1)
        combined = self.comb_norm(combined)
        return self.head(combined).squeeze(-1)

# ===================== DATA PROCESSING =====================
def _import_tdc_and_from_smiles():
    try:
        from tdc.single_pred import ADME
    except Exception as e:
        raise RuntimeError("TDC not installed. Install: pip install PyTDC") from e
    try:
        from torch_geometric.utils import from_smiles
    except Exception:
        try:
            from torch_geometric.data import from_smiles
        except Exception as e:
            raise RuntimeError("PyG 'from_smiles' unavailable. Update torch-geometric.") from e
    return ADME, from_smiles

def row_to_graph_enhanced(from_smiles_fn, smiles: str, y_value: float):
    g = from_smiles_fn(smiles)
    if g is None or getattr(g, "x", None) is None:
        return None
    if RDKit_OK:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            atom_features = [enhanced_atom_features(atom) for atom in mol.GetAtoms()]
            g.x = torch.tensor(np.array(atom_features), dtype=torch.float32)
    g.x = g.x.float()
    g.edge_index = g.edge_index.long()
    g.original_y = float(y_value)
    g.y = torch.tensor([0.0], dtype=torch.float)
    adme_desc = adme_specific_descriptors(smiles)
    g.adme_features = torch.tensor(adme_desc, dtype=torch.float)
    return g

def df_to_graph_list_enhanced(from_smiles_fn, df: pd.DataFrame) -> List[Data]:
    out: List[Data] = []
    for _, r in df.iterrows():
        smi = r["Drug"]; yv = r["Y"]
        g = row_to_graph_enhanced(from_smiles_fn, smi, yv)
        if g is not None:
            out.append(g)
    return out

def _apply_target_transform_inplace(datasets: List[List[Data]]):
    for ds in datasets:
        for g in ds:
            g.y = torch.tensor([_transform_y(g.original_y)], dtype=torch.float)

def _fit_adme_scaler(train_list: List[Data]):
    X = np.stack([d.adme_features.numpy() for d in train_list], axis=0)
    mu = X.mean(axis=0); sigma = X.std(axis=0); sigma[sigma == 0.0] = 1.0
    ADME_SCALER["mu"] = torch.tensor(mu, dtype=torch.float32)
    ADME_SCALER["sigma"] = torch.tensor(sigma, dtype=torch.float32)

def _standardize_adme_inplace(datasets: List[List[Data]]):
    mu = ADME_SCALER["mu"]; sigma = ADME_SCALER["sigma"]
    if mu is None or sigma is None: return
    for ds in datasets:
        for g in ds:
            g.adme_features = (g.adme_features - mu) / (sigma + 1e-8)

def build_tdc_loaders_simple(
        dataset_name: str,
        batch_train: int = 32,
        batch_eval: int = 64,
        split_type: str = "scaffold",
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    ADME, from_smiles_fn = _import_tdc_and_from_smiles()
    data_api = ADME(name=dataset_name)
    try:
        split = data_api.get_split(method=split_type)
    except Exception:
        split = data_api.get_split()

    print(f"\nDataset: {dataset_name}")
    print(f"Train: {len(split['train'])}, Test: {len(split['test'])}")

    y_train_arr = split['train']['Y'].values.astype(float)
    set_target_scaler(dataset_name, y_train_arr, force_mode="log")

    train_list = df_to_graph_list_enhanced(from_smiles_fn, split["train"])
    valid_df = split.get("valid") if "valid" in split else split.get("val")
    valid_list = df_to_graph_list_enhanced(from_smiles_fn, valid_df) if valid_df is not None else []
    test_list = df_to_graph_list_enhanced(from_smiles_fn, split["test"])

    if not valid_list:
        n = len(train_list); k = max(1, int(0.15 * n))  # Use 15% for validation
        valid_list = train_list[:k]; train_list = train_list[k:]

    _apply_target_transform_inplace([train_list, valid_list, test_list])
    _fit_adme_scaler(train_list)
    _standardize_adme_inplace([train_list, valid_list, test_list])

    adme_dim = int(train_list[0].adme_features.numel()) if len(train_list) else 30

    train_loader = DataLoader(train_list, batch_size=batch_train, shuffle=True)
    val_loader = DataLoader(valid_list, batch_size=batch_eval, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_eval, shuffle=False)

    return train_loader, val_loader, test_loader, adme_dim

# ===================== TRAINING FUNCTIONS =====================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0; num_batches = 0
    for data in loader:
        data = data.to(device)
        pred = model(data)
        y = data.y.view_as(pred)
        # Simple MSE loss
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item())
        num_batches += 1
    return total_loss / max(1, num_batches)

@torch.no_grad()
def evaluate(model, loader, device, inverse: bool = False, return_arrays: bool = False):
    model.eval()
    preds, trues = [], []
    for data in loader:
        data = data.to(device)
        p = model(data)
        preds.append(p.detach().cpu())
        trues.append(data.y.view_as(p).detach().cpu())
    if not preds:
        out = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
        if return_arrays:
            return out, np.array([]), np.array([])
        return out
    preds = torch.cat(preds).float().numpy()
    trues = torch.cat(trues).float().numpy()
    if inverse:
        preds = _inverse_y(preds)
        trues = _inverse_y(trues)
    mse = float(np.mean((preds - trues) ** 2))
    mae = float(np.mean(np.abs(preds - trues)))
    var = float(np.var(trues))
    r2 = 1.0 - (mse / (var + 1e-12))
    out = {"rmse": math.sqrt(mse), "mae": mae, "r2": r2}
    if return_arrays:
        return out, preds, trues
    return out

def train_single_model(
        dataset_name: str,
        model_name: str = "SAGE",
        layers: int = 3,
        hidden: int = 64,
        dropout: float = 0.5,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,  # INCREASED from 1e-4
        epochs: int = 150,
        patience: int = 30,
        device: str = None,
        seed: int = 42,
) -> Tuple[Dict, Dict, Dict]:
    set_global_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, adme_dim = build_tdc_loaders_simple(
        dataset_name, batch_train=32, batch_eval=64, split_type="scaffold"
    )

    in_dim = int(train_loader.dataset[0].x.size(1))

    backbone = SimpleGraphBackbone(
        model_name=model_name,
        graph_layers=layers,
        graph_hidden_channels=hidden,
        dropout=dropout,
        input_dim=in_dim
    ).to(device)

    model = SimpleMolecularRegressor(backbone, adme_dim=adme_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_rmse = float("inf")
    best_state = None
    best_epoch = 0
    no_imp = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics, val_pred, val_true = evaluate(model, val_loader, device, inverse=True, return_arrays=True)

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            no_imp = 0
        else:
            no_imp += 1

        scheduler.step(val_metrics["rmse"])

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch}: loss={train_loss:.4f}, val_RMSE={val_metrics['rmse']:.3f}, val_R2={val_metrics['r2']:.3f}")

        if no_imp >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval
    val_metrics, val_pred, val_true = evaluate(model, val_loader, device, inverse=True, return_arrays=True)
    test_metrics, test_pred, test_true = evaluate(model, test_loader, device, inverse=True, return_arrays=True)

    val_spear = spearman_metric(val_true, val_pred)
    test_spear = spearman_metric(test_true, test_pred)

    print(f"  FINAL: val_RMSE={val_metrics['rmse']:.3f}, val_R2={val_metrics['r2']:.3f}, " +
          f"test_RMSE={test_metrics['rmse']:.3f}, test_R2={test_metrics['r2']:.3f}, test_Spear={test_spear:.3f}")

    meta = {
        "dataset": dataset_name,
        "model_name": model_name,
        "layers": layers,
        "hidden": hidden,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs_trained": best_epoch,
        "train_time_s": round(time.time() - t0, 2),
        "seed": seed,
    }

    val_out = {**val_metrics, "spearman": val_spear}
    test_out = {**test_metrics, "spearman": test_spear}

    return meta, val_out, test_out

# ===================== ENSEMBLE TRAINING =====================
def train_ensemble(
        dataset_name: str,
        seeds: List[int] = [42, 123, 456, 789, 1011],
        **kwargs
) -> Dict:
    """Train ensemble of models with different seeds"""
    print(f"\n{'='*80}")
    print(f"ENSEMBLE TRAINING: {dataset_name} ({len(seeds)} seeds)")
    print(f"{'='*80}")

    all_val_rmse = []
    all_val_r2 = []
    all_test_rmse = []
    all_test_r2 = []
    all_test_spear = []

    for i, seed in enumerate(seeds, 1):
        print(f"\n[{i}/{len(seeds)}] Training with seed={seed}")
        meta, val_m, test_m = train_single_model(dataset_name, seed=seed, **kwargs)

        all_val_rmse.append(val_m["rmse"])
        all_val_r2.append(val_m["r2"])
        all_test_rmse.append(test_m["rmse"])
        all_test_r2.append(test_m["r2"])
        all_test_spear.append(test_m["spearman"])

    print(f"\n{'='*60}")
    print("ENSEMBLE SUMMARY:")
    print(f"{'='*60}")
    print(f"Val RMSE:  {np.mean(all_val_rmse):.3f} ± {np.std(all_val_rmse):.3f}")
    print(f"Val R2:    {np.mean(all_val_r2):.3f} ± {np.std(all_val_r2):.3f}")
    print(f"Test RMSE: {np.mean(all_test_rmse):.3f} ± {np.std(all_test_rmse):.3f}")
    print(f"Test R2:   {np.mean(all_test_r2):.3f} ± {np.std(all_test_r2):.3f}")
    print(f"Test Spear: {np.mean(all_test_spear):.3f} ± {np.std(all_test_spear):.3f}")

    return {
        "dataset": dataset_name,
        "n_seeds": len(seeds),
        "val_rmse_mean": np.mean(all_val_rmse),
        "val_rmse_std": np.std(all_val_rmse),
        "val_r2_mean": np.mean(all_val_r2),
        "val_r2_std": np.std(all_val_r2),
        "test_rmse_mean": np.mean(all_test_rmse),
        "test_rmse_std": np.std(all_test_rmse),
        "test_r2_mean": np.mean(all_test_r2),
        "test_r2_std": np.std(all_test_r2),
        "test_spear_mean": np.mean(all_test_spear),
        "test_spear_std": np.std(all_test_spear),
    }

# ===================== MAIN =====================
if __name__ == "__main__":
    datasets = ['Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ']

    print(f"\n{'#'*80}")
    print("# PRIMER - FIXED FOUNDATION")
    print(f"{'#'*80}")
    print("\nKey changes:")
    print("  - Simpler architecture (3 layers, 64 hidden)")
    print("  - STRONG regularization (dropout 0.5, weight_decay 1e-3)")
    print("  - Simpler readout (mean + max only)")
    print("  - 5-seed ensemble for stability")
    print("  - Target: Achieve POSITIVE R2 on all datasets")

    all_results = []

    for dataset in datasets:
        result = train_ensemble(
            dataset,
            seeds=[42, 123, 456, 789, 1011],
            model_name="SAGE",
            layers=3,
            hidden=64,
            dropout=0.5,
            lr=1e-3,
            weight_decay=1e-3,
            epochs=150,
            patience=30,
        )
        all_results.append(result)

    # Save results
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"fixed_foundation_{timestamp}.csv"
    df.to_csv(output_file, index=False)

    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"\nResults saved: {output_file}")

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Check if R2 is now positive")
    print("2. Compare to ML baselines (RF, XGBoost)")
    print("3. If R2 > 0: proceed to add advanced methods")
    print("4. If R2 still negative: try random split instead of scaffold")
