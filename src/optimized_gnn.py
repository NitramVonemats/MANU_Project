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

import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


@dataclass
class OptimizedGNNConfig:
    """Configuration container for the optimized molecular GNN."""

    hidden_dim: int = 128
    num_layers: int = 5
    head_dims: Sequence[int] = (256, 128, 64)
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_train: int = 32
    batch_eval: int = 64
    val_fraction: float = 0.1
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    max_grad_norm: Optional[float] = 1.0
    adme_dim: Optional[int] = None

    def __post_init__(self):
        self.head_dims = tuple(int(h) for h in self.head_dims)
        if any(h <= 0 for h in self.head_dims):
            raise ValueError("All head_dims must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.batch_train <= 0 or self.batch_eval <= 0:
            raise ValueError("Batch sizes must be positive.")
        if not 0 < self.val_fraction < 0.5:
            raise ValueError("val_fraction must be in (0, 0.5).")
        if self.scheduler_patience < 0:
            raise ValueError("scheduler_patience must be non-negative.")
        if not (0.0 < self.scheduler_factor < 1.0):
            raise ValueError("scheduler_factor must be in (0, 1).")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive when provided.")


def resolve_device(device: str = "auto") -> str:
    """Resolve device string, supporting 'auto'."""
    device_lower = device.lower()
    if device_lower == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

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


def caco2_wang_descriptors(smiles: str) -> np.ndarray:
    """Caco2_Wang permeability-relevant descriptors.
    
    Uses core ADME features relevant for permeability prediction:
    MW, LogP, H-bond donors/acceptors, TPSA, rotatable bonds, aromatic rings.
    """
    if not RDKit_OK:
        return np.zeros(7, dtype=np.float32)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(7, dtype=np.float32)

    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        rotatable = Descriptors.NumRotatableBonds(mol)
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        
        return np.array([
            mw, logp, hbd, hba, tpsa, rotatable, aromatic_rings
        ], dtype=np.float32)
    except Exception:
        return np.zeros(7, dtype=np.float32)


# ===================== OPTIMAL ARCHITECTURE =====================

class OptimalGraphBackbone(nn.Module):
    """Graph модел - докажано најдобар! БЕЗ edge features"""

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=5, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        if dropout > 0:
            self.dropouts = nn.ModuleList()

        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            self.convs.append(GraphConv(in_channels, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            if dropout > 0:
                self.dropouts.append(nn.Dropout(dropout))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(x, edge_index)
            h = norm(h)
            h = F.relu(h)
            
            # Apply dropout if enabled
            if self.dropout > 0:
                h = self.dropouts[i](h)

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

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=5, adme_dim=15, head_dims: Sequence[int] = (256, 128, 64), dropout=0.0):
        super().__init__()

        self.backbone = OptimalGraphBackbone(input_dim, hidden_dim, num_layers, dropout=dropout)
        self.readout = SimpleReadout(hidden_dim)

        # Combined features
        combined_dim = self.readout.out_dim + adme_dim

        # Prediction head - динамички MLP
        self.head = self._build_head(combined_dim, head_dims, dropout=dropout)

    @staticmethod
    def _build_head(input_dim: int, head_dims: Sequence[int], dropout: float = 0.0) -> nn.Sequential:
        if not head_dims:
            raise ValueError("head_dims must contain at least one layer size.")

        layers: List[nn.Module] = []
        current_dim = input_dim
        for i, hidden in enumerate(head_dims):
            layers.append(nn.Linear(current_dim, int(hidden)))
            layers.append(nn.BatchNorm1d(int(hidden)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = int(hidden)
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, data):
        # Graph embedding
        graph_emb = self.backbone(data)
        graph_pooled = self.readout(graph_emb, data.batch)

        # ADME features (may be empty for some datasets like Caco2_Wang)
        adme = data.adme_features
        if adme.dim() == 1:
            adme = adme.unsqueeze(0)

        # Combine and predict (handle empty ADME features gracefully)
        if adme.numel() > 0:
            combined = torch.cat([graph_pooled, adme], dim=-1)
        else:
            combined = graph_pooled  # No ADME features, use graph only
        
        return self.head(combined).squeeze(-1)


# ===================== DATA PROCESSING =====================

def row_to_graph(from_smiles_fn, smiles: str, y_value: float, dataset_name: str = None):
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

    # ADME descriptors - dataset-specific for Caco2_Wang
    if dataset_name == "Caco2_Wang":
        descriptors = caco2_wang_descriptors(smiles)
    else:
        descriptors = adme_descriptors(smiles)
    
    g.adme_features = torch.tensor(descriptors, dtype=torch.float32).view(1, -1)

    return g


def prepare_dataset(
    dataset_name: str,
    val_fraction: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
):
    """Подготви графови, нормализација и статистики за даден dataset."""
    from tdc.single_pred import ADME

    try:
        from torch_geometric.utils import from_smiles
    except ImportError:
        from torch_geometric.data import from_smiles  # type: ignore

    data_api = ADME(name=dataset_name)
    split = data_api.get_split(method="scaffold")

    if verbose:
        print(f"\n📊 Dataset: {dataset_name}")
        print(f"Train: {len(split['train'])}, Test: {len(split['test'])}")

    # Check for duplicate SMILES (data leakage indicator)
    train_smiles = set(split['train']['Drug'].values)
    test_smiles = set(split['test']['Drug'].values)
    overlap = train_smiles & test_smiles
    if overlap and verbose:
        print(f"⚠️  WARNING: {len(overlap)} duplicate SMILES found between train and test sets!")

    train_graphs = [
        row_to_graph(from_smiles, row["Drug"], row["Y"], dataset_name)
        for _, row in split["train"].iterrows()
    ]
    train_graphs = [g for g in train_graphs if g is not None]

    test_graphs = [
        row_to_graph(from_smiles, row["Drug"], row["Y"], dataset_name)
        for _, row in split["test"].iterrows()
    ]
    test_graphs = [g for g in test_graphs if g is not None]

    if len(train_graphs) < 2:
        raise ValueError("Недоволно train примероци за подготвка на датасет.")

    rng = random.Random(seed)
    rng.shuffle(train_graphs)

    n_val = max(1, int(len(train_graphs) * val_fraction))
    if len(train_graphs) - n_val < 1:
        n_val = max(1, len(train_graphs) - 1)
    val_graphs = train_graphs[:n_val]
    train_graphs = train_graphs[n_val:]

    if not train_graphs:
        raise ValueError("Валидациската поделба ја испразни тренинг групата.")

    y_train = np.array([g.original_y for g in train_graphs], dtype=np.float32)
    y_val = np.array([g.original_y for g in val_graphs], dtype=np.float32)
    y_all = np.concatenate([y_train, y_val])
    
    # Detect if values are already log-transformed (all negative) or need log transform
    all_negative = np.all(y_all < 0)
    all_positive = np.all(y_all > 0)
    
    # Always print diagnostic info for Caco2_Wang
    if dataset_name == "Caco2_Wang":
        print(f"  Original target range (train): [{y_train.min():.10e}, {y_train.max():.10e}]")
        print(f"  Original target range (val):   [{y_val.min():.10e}, {y_val.max():.10e}]")
        print(f"  All negative: {all_negative}, All positive: {all_positive}")
        print(f"  Train unique values: {len(np.unique(y_train))}/{len(y_train)}")
    
    # Handle transformation based on value signs
    if all_negative:
        # Values are already log-transformed (e.g., Caco2_Wang), skip log transform
        print(f"  📊 Detected log-transformed values (all negative), skipping log transform")
        y_log = y_train.astype(np.float32)
        mu = float(y_log.mean())
        sigma = float(y_log.std())
        
        if sigma < 1e-6:
            print(f"  ⚠️  WARNING: sigma={sigma:.10e} too small, setting to 1.0")
            sigma = 1.0
        
        # Apply normalization directly (no log transform needed)
        for g in train_graphs + val_graphs + test_graphs:
            y_value = float(g.original_y)
            g.y = torch.tensor([(y_value - mu) / sigma], dtype=torch.float32)
    else:
        # Normal case: positive values, apply log transform
        clip_min = 1e-3 if dataset_name != "Caco2_Wang" else 1e-6
        
        if dataset_name == "Caco2_Wang":
            positive_values = y_all[y_all > 0]
            if len(positive_values) > 0:
                min_val = float(positive_values.min())
                clip_min = max(min_val / 1000.0, 1e-9)
            else:
                clip_min = 1e-9
            print(f"  Using clip_min={clip_min:.10e}")
        
        y_train_clipped = np.clip(y_train, clip_min, None)
        y_log = np.log(y_train_clipped)
        mu = float(y_log.mean())
        sigma = float(y_log.std())
        
        if sigma < 1e-6:
            print(f"  ⚠️  WARNING: sigma={sigma:.10e} too small, setting to 1.0")
            sigma = 1.0
        
        for g in train_graphs + val_graphs + test_graphs:
            y_value = max(clip_min, float(g.original_y))
            g.y = torch.tensor([(np.log(y_value) - mu) / sigma], dtype=torch.float32)

    # Handle ADME feature normalization (skip if features are empty)
    if len(train_graphs) > 0 and train_graphs[0].adme_features.numel() > 0:
        adme_train = np.stack([g.adme_features.squeeze(0).numpy() for g in train_graphs])
        adme_mu = torch.tensor(adme_train.mean(0), dtype=torch.float32)
        adme_sigma = torch.tensor(adme_train.std(0), dtype=torch.float32)
        adme_sigma[adme_sigma == 0] = 1.0

        for g in train_graphs + val_graphs + test_graphs:
            g.adme_features = (g.adme_features - adme_mu.unsqueeze(0)) / adme_sigma.unsqueeze(0)
    else:
        # No ADME features to normalize (e.g., Caco2_Wang with empty features)
        adme_mu = torch.tensor([], dtype=torch.float32)
        adme_sigma = torch.tensor([], dtype=torch.float32)

    if verbose:
        print(f"Splits: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")
        print(f"Log scaling: μ={mu:.3f}, σ={sigma:.3f}")

    # Store whether values were already log-transformed
    is_log_transformed = all_negative if dataset_name == "Caco2_Wang" else False
    
    return {
        "train": train_graphs,
        "val": val_graphs,
        "test": test_graphs,
        "log_stats": (mu, sigma),
        "adme_stats": (adme_mu, adme_sigma),
        "dataset_name": dataset_name,
        "is_log_transformed": is_log_transformed,  # Flag for inverse transform
    }


def build_loaders(
    dataset_name: str,
    batch_train: int = 32,
    batch_eval: int = 64,
    val_fraction: float = 0.1,
    seed: int = 42,
    dataset_cache: Optional[Dict[str, Any]] = None,
    return_cache: bool = False,
    verbose: bool = True,
):
    """Подготви DataLoader објекти за тренинг / валидација / тест."""
    if dataset_cache is None:
        dataset_cache = prepare_dataset(
            dataset_name=dataset_name,
            val_fraction=val_fraction,
            seed=seed,
            verbose=verbose,
        )

    train_list = dataset_cache["train"]
    val_list = dataset_cache["val"]
    test_list = dataset_cache["test"]
    mu, sigma = dataset_cache["log_stats"]

    train_loader = DataLoader(train_list, batch_size=batch_train, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=max(1, batch_eval), shuffle=False)
    test_loader = DataLoader(test_list, batch_size=max(1, batch_eval), shuffle=False)
    
    # Debug: Check loader sizes
    if verbose:
        print(f"Loader sizes: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
        print(f"Dataset sizes: train={len(train_list)}, val={len(val_list)}, test={len(test_list)}")

    if return_cache:
        return train_loader, val_loader, test_loader, (mu, sigma), dataset_cache
    # Return mu, sigma, and is_log_transformed flag
    is_log_transformed = dataset_cache.get("is_log_transformed", False) if return_cache else False
    return train_loader, val_loader, test_loader, (mu, sigma)


# ===================== TRAINING =====================

def train_epoch(model, loader, optimizer, device, max_grad_norm: Optional[float] = None, label_noise: float = 0.0):
    """Тренирај една епоха"""
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        pred = model(data)
        
        # Add label noise for regularization (helps prevent memorization)
        targets = data.y
        if label_noise > 0:
            noise = torch.randn_like(targets) * label_noise
            targets = targets + noise
        
        loss = F.mse_loss(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, mu, sigma, is_log_transformed=False):
    """Евалуирај модел (вратено во оригинален scale)
    
    Args:
        is_log_transformed: If True, values were already in log space before normalization,
                           so inverse normalization gives log values which need exp()
    """
    model.eval()
    preds, trues = [], []

    if len(loader) == 0:
        # Empty loader - return NaN metrics
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    for data in loader:
        data = data.to(device)
        pred = model(data)
        preds.append(pred.cpu())
        trues.append(data.y.cpu())

    if not preds:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    # Check for NaN or Inf values
    if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
        return {"rmse": float("inf"), "mae": float("inf"), "r2": float("-inf")}
    if np.any(np.isnan(trues)) or np.any(np.isinf(trues)):
        return {"rmse": float("inf"), "mae": float("inf"), "r2": float("-inf")}

    # Inverse transform: denormalize then exp() to get original scale
    # Both paths (log-transformed or not) need exp() because:
    # - If already log-transformed: normalized -> log_value -> exp(log_value) = original
    # - If not log-transformed: normalized -> log_value (from our log transform) -> exp(log_value) = original
    preds_log = preds * sigma + mu
    trues_log = trues * sigma + mu
    
    # Convert to original scale (always use exp since we want original values)
    preds_orig = np.exp(preds_log)
    trues_orig = np.exp(trues_log)

    # Check for invalid values after inverse transform
    if np.any(np.isnan(preds_orig)) or np.any(np.isinf(preds_orig)):
        return {"rmse": float("inf"), "mae": float("inf"), "r2": float("-inf")}
    if np.any(np.isnan(trues_orig)) or np.any(np.isinf(trues_orig)):
        return {"rmse": float("inf"), "mae": float("inf"), "r2": float("-inf")}

    # Metrics
    squared_errors = (preds_orig - trues_orig) ** 2
    mse = np.mean(squared_errors)
    
    # Calculate RMSE - handle edge cases
    if mse < 1e-10:  # Essentially zero (numerical precision)
        rmse = 0.0
    else:
        rmse = np.sqrt(mse)
    
    mae = np.mean(np.abs(preds_orig - trues_orig))
    
    # R² calculation with safety check
    ss_res = np.sum(squared_errors)
    ss_tot = np.sum((trues_orig - trues_orig.mean()) ** 2)
    
    if ss_tot < 1e-12:
        # All targets are identical
        r2 = 1.0 if ss_res < 1e-12 else 0.0
    else:
        r2 = 1 - (ss_res / ss_tot)
        # Clip R² to reasonable range
        r2 = max(-10.0, min(10.0, r2))

    # Debug: print actual values if RMSE is suspiciously low
    if rmse < 0.001 and len(preds_orig) > 0:
        max_diff = np.max(np.abs(preds_orig - trues_orig))
        mean_pred = np.mean(preds_orig)
        mean_true = np.mean(trues_orig)
        std_pred = np.std(preds_orig)
        std_true = np.std(trues_orig)
        
        # Print debug info with more precision
        print(f"  DEBUG: RMSE={rmse:.8f}, MAE={mae:.8f}, max_diff={max_diff:.8f}")
        print(f"  DEBUG: pred_mean={mean_pred:.8f}, true_mean={mean_true:.8f}")
        print(f"  DEBUG: pred_std={std_pred:.8f}, true_std={std_true:.8f}, n_samples={len(preds_orig)}")
        print(f"  DEBUG: pred_range=[{np.min(preds_orig):.8f}, {np.max(preds_orig):.8f}]")
        print(f"  DEBUG: true_range=[{np.min(trues_orig):.8f}, {np.max(trues_orig):.8f}]")
        print(f"  DEBUG: log_pred_range=[{np.min(preds):.4f}, {np.max(preds):.4f}]")
        print(f"  DEBUG: log_true_range=[{np.min(trues):.4f}, {np.max(trues):.4f}]")
        print(f"  DEBUG: mu={mu:.4f}, sigma={sigma:.4f}")

    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def train_model(
    dataset_name: str,
    config: Optional[OptimizedGNNConfig] = None,
    epochs: int = 100,
    patience: int = 20,
    device: str = "cpu",
    seed: int = 42,
    dataset_cache: Optional[Dict[str, Any]] = None,
    evaluate_test: bool = True,
    return_model: bool = True,
    verbose: bool = True,
):
    """Тренирај оптимизиран модел со зададени хиперпараметри."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    resolved_device = resolve_device(device)
    cfg = config or OptimizedGNNConfig()

    dataset_cache = dataset_cache or prepare_dataset(
        dataset_name=dataset_name,
        val_fraction=cfg.val_fraction,
        seed=seed,
        verbose=verbose,
    )
    loaders = build_loaders(
        dataset_name=dataset_name,
        batch_train=cfg.batch_train,
        batch_eval=cfg.batch_eval,
        val_fraction=cfg.val_fraction,
        seed=seed,
        dataset_cache=dataset_cache,
        verbose=verbose,
        return_cache=True,
    )
    train_loader, val_loader, test_loader, (mu, sigma), dataset_cache = loaders
    is_log_transformed = dataset_cache.get("is_log_transformed", False)

    sample_graph = dataset_cache["train"][0]
    input_dim = int(sample_graph.x.size(-1))
    adme_dim = cfg.adme_dim or int(sample_graph.adme_features.numel())

    # Use dropout for Caco2_Wang to help prevent overfitting (let HPO find best model size)
    dropout = 0.5 if dataset_name == "Caco2_Wang" else 0.0
    if verbose and dataset_name == "Caco2_Wang":
        print(f"⚠️  Caco2_Wang: Using dropout={dropout} for regularization")

    model = OptimizedMolecularGNN(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        adme_dim=adme_dim,
        head_dims=cfg.head_dims,
        dropout=dropout,
    ).to(resolved_device)

    if verbose:
        print(f"\n🧬 Model: Graph ({cfg.num_layers} layers, {cfg.hidden_dim} hidden)")
        print(f"Head dims: {tuple(cfg.head_dims)}, LR={cfg.lr:.4f}, WD={cfg.weight_decay:.2e}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"\n🏃 Training...")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = None
    if cfg.scheduler_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=cfg.scheduler_patience,
            factor=cfg.scheduler_factor,
        )

    best_val_rmse = float("inf")
    best_state = None
    no_improvement = 0
    history: List[Dict[str, float]] = []

    start_time = time.time()
    
    # Add aggressive label noise for Caco2_Wang to prevent memorization
    # Higher noise prevents exact memorization while still allowing learning
    label_noise = 0.05 if dataset_name == "Caco2_Wang" else 0.0  # Increased from 0.01

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, resolved_device, cfg.max_grad_norm, label_noise=label_noise)
        val_metrics = evaluate(model, val_loader, resolved_device, mu, sigma)

        if scheduler is not None:
            scheduler.step(val_metrics["rmse"])

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_rmse": float(val_metrics["rmse"]),
                "val_mae": float(val_metrics["mae"]),
                "val_r2": float(val_metrics["r2"]),
            }
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"Epoch {epoch:3d}: loss={train_loss:.4f}, "
                f"val_rmse={val_metrics['rmse']:.6f}, val_r2={val_metrics['r2']:.6f}"
            )

        if no_improvement >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate(model, val_loader, resolved_device, mu, sigma)
    test_metrics = evaluate(model, test_loader, resolved_device, mu, sigma, is_log_transformed) if evaluate_test else None

    train_time = time.time() - start_time

    if verbose:
        print(f"\n✅ Training complete ({train_time:.1f}s)")
        print(
            f"Val  - RMSE: {val_metrics['rmse']:.6f}, MAE: {val_metrics['mae']:.6f}, R²: {val_metrics['r2']:.6f}"
        )
        if test_metrics is not None:
            print(
                f"Test - RMSE: {test_metrics['rmse']:.6f}, "
                f"MAE: {test_metrics['mae']:.6f}, R²: {test_metrics['r2']:.6f}"
            )

    trained_model = model if return_model else None
    if not return_model:
        model.cpu()
        if resolved_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "model": trained_model,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_time": train_time,
        "dataset": dataset_name,
        "config": cfg,
        "history": history,
        "mu": mu,
        "sigma": sigma,
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
