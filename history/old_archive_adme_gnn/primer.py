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
from torch_geometric.nn import (
    GCNConv, SAGEConv, GINConv, GATConv, TAGConv, ChebConv, SGConv,
    TransformerConv, GraphConv, global_mean_pool, global_max_pool,
    GlobalAttention
)
import random

# Optional / version-safe imports for PyG norms
try:
    from torch_geometric.nn import LayerNorm as PyGLayerNorm
except Exception:
    PyGLayerNorm = None

try:
    from torch_geometric.nn import GraphNorm as PyGGraphNorm
except Exception:
    PyGGraphNorm = None

# ===================== RDKit SETUP =====================
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKit_OK = True
except Exception:
    RDKit_OK = False

# ===================== GLOBAL SCALERS (FIX) =====================
TARGET_SCALER = {"mode": "log", "mu": 0.0, "sigma": 1.0}  # per-dataset target scaling
ADME_SCALER = {"mu": None, "sigma": None}                 # ADME standardization (train-only)

RNG_SEED = 42

# ===================== UTILS =====================
# Spearman: try scipy, else numpy ranks fallback
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

# ---------- Target scaling helpers (FIX) ----------
def set_target_scaler(dataset_name: str, y_train: np.ndarray, force_mode: str = "log"):
    mode = force_mode  # "log" or "standard"
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
        z = np.clip(z, -10, 10)  # numeric safety
        return np.exp(z)
    return t * TARGET_SCALER["sigma"] + TARGET_SCALER["mu"]

# ===================== ENHANCED MOLECULAR FEATURES =====================
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

def enhanced_bond_features(bond):
    try:
        features = [
            float(bond.GetBondTypeAsDouble()),
            int(bond.GetBondType() == Chem.BondType.SINGLE),
            int(bond.GetBondType() == Chem.BondType.DOUBLE),
            int(bond.GetBondType() == Chem.BondType.TRIPLE),
            int(bond.GetBondType() == Chem.BondType.AROMATIC),
            int(bond.GetIsAromatic()), int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            int(bond.GetStereo() == Chem.BondStereo.STEREONONE),
            int(bond.GetStereo() == Chem.BondStereo.STEREOANY),
            int(bond.GetStereo() == Chem.BondStereo.STEREOZ),
            int(bond.GetStereo() == Chem.BondStereo.STEREOE),
        ]
        return np.array(features, dtype=np.float32)
    except Exception:
        return np.array([1.0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32)

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

# ===================== FIXED ARCHITECTURE =====================
class GraphBackbone(nn.Module):
    def __init__(self,
                 model_name="GCN",
                 graph_layers=3,
                 graph_hidden_channels=64,
                 graph_norm="BatchNorm",
                 activation="relu",
                 heads=1,
                 aggr="mean",
                 residual=True,
                 use_edge_features=True,
                 dropout=0.2,
                 input_dim=37,
                 **kwargs):
        super().__init__()
        self.model_name = model_name
        self.graph_layers = graph_layers
        self.output_dim = graph_hidden_channels
        self.residual = residual
        self.use_edge_features = use_edge_features

        convs, norms, dropouts = [], [], []
        edge_dim = 12 if use_edge_features else None

        for i in range(graph_layers):
            input_channels = input_dim if i == 0 else graph_hidden_channels

            if model_name == "GCN":
                conv = GCNConv(input_channels, graph_hidden_channels)
            elif model_name == "SAGE":
                conv = SAGEConv(input_channels, graph_hidden_channels, aggr=aggr)
            elif model_name == "GIN":
                nn_lin = nn.Sequential(
                    nn.Linear(input_channels, graph_hidden_channels),
                    nn.ReLU(),
                    nn.Linear(graph_hidden_channels, graph_hidden_channels)
                )
                conv = GINConv(nn_lin)
            elif model_name == "GAT":
                conv = GATConv(input_channels, graph_hidden_channels // max(1, heads),
                               heads=heads, edge_dim=edge_dim if use_edge_features else None)
            elif model_name == "TAG":
                conv = TAGConv(input_channels, graph_hidden_channels)
            elif model_name == "Cheb":
                conv = ChebConv(input_channels, graph_hidden_channels, K=kwargs.get("K", 2))
            elif model_name == "SGC":
                conv = SGConv(input_channels, graph_hidden_channels, K=kwargs.get("K", 2))
            elif model_name == "Transformer":
                conv = TransformerConv(input_channels, graph_hidden_channels // max(1, heads),
                                       heads=heads, edge_dim=edge_dim if use_edge_features else None)
            else:  # GraphConv
                conv = GraphConv(input_channels, graph_hidden_channels)

            convs.append(conv)

            if graph_norm == "BatchNorm":
                norms.append(nn.BatchNorm1d(graph_hidden_channels))
            elif graph_norm == "GraphNorm" and PyGGraphNorm is not None:
                norms.append(PyGGraphNorm(graph_hidden_channels))
            elif graph_norm == "LayerNorm":
                if PyGLayerNorm is not None:
                    norms.append(PyGLayerNorm(graph_hidden_channels, mode="node"))
                else:
                    norms.append(nn.LayerNorm(graph_hidden_channels))
            else:
                norms.append(nn.BatchNorm1d(graph_hidden_channels))

            dropouts.append(nn.Dropout(dropout))

        self._convs = nn.ModuleList(convs)
        self._norms = nn.ModuleList(norms)
        self._dropouts = nn.ModuleList(dropouts)

        if activation == "swish":
            self._act = lambda x: x * torch.sigmoid(x)
        elif activation == "gelu":
            self._act = F.gelu
        elif activation == "leaky_relu":
            self._act = F.leaky_relu
        else:
            self._act = F.relu

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None) if self.use_edge_features else None

        for (conv, norm, dropout) in zip(self._convs, self._norms, self._dropouts):
            if hasattr(conv, 'edge_dim') and edge_attr is not None and self.model_name in ["GAT", "Transformer"]:
                h = conv(x, edge_index, edge_attr)
            else:
                h = conv(x, edge_index)
            h = norm(h)
            h = self._act(h)
            h = dropout(h)
            if self.residual and h.shape == x.shape:
                x = x + h
            else:
                x = h
        return x

class MultiScaleReadout(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.attn = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(in_dim, max(32, in_dim // 2)),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(max(32, in_dim // 2), 1),
            )
        )
        self.local_pool = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
        )
        self.out_dim = (in_dim * 3) + (in_dim // 2)

    def forward(self, x, batch):
        mean_p = global_mean_pool(x, batch)
        max_p = global_max_pool(x, batch)
        att_p = self.attn(x, batch)
        local_x = self.local_pool(x)
        local_mean = global_mean_pool(local_x, batch)
        return torch.cat([mean_p, max_p, att_p, local_mean], dim=-1)

class ImprovedMolecularRegressor(nn.Module):
    def __init__(self, backbone: GraphBackbone, adme_dim: int = 30):
        super().__init__()
        self.backbone = backbone
        self.adme_dim = adme_dim
        self.readout = MultiScaleReadout(backbone.output_dim)
        combined_dim = self.readout.out_dim + adme_dim
        self.comb_norm = nn.BatchNorm1d(combined_dim)
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data: Data) -> torch.Tensor:
        graph_emb = self.backbone(data)
        graph_pooled = self.readout(graph_emb, data.batch)
        adme = getattr(data, "adme_features", None)
        if adme is None:
            batch_size = data.batch.max() + 1 if hasattr(data, 'batch') else 1
            adme = torch.zeros(batch_size, self.adme_dim, device=data.x.device)
        elif adme.dim() == 3 and adme.size(1) == 1:
            adme = adme.squeeze(1)
        elif adme.dim() == 1:
            adme = adme.view(-1, self.adme_dim)
        combined = torch.cat([graph_pooled, adme], dim=-1)
        combined = self.comb_norm(combined)
        return self.head(combined).squeeze(-1)

class RobustLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        l2 = F.mse_loss(pred, target)
        huber = F.smooth_l1_loss(pred, target)
        return self.alpha * l2 + (1 - self.alpha) * l1 + 0.1 * huber

# ===================== ENHANCED PARAMETER GENERATOR =====================
class EnhancedGNNTester:
    def __init__(self, train_data=None):
        self.train_data = train_data
    def create_enhanced_parameter_combinations(self, max_combinations=50):
        model_pool = [
            ("SAGE", 0.30), ("TAG", 0.20), ("SGC", 0.20), ("GIN", 0.15),
            ("GCN", 0.10), ("Graph", 0.05)
        ]
        models = [m for m, _ in model_pool]
        probs = [p for _, p in model_pool]
        layers = [2, 3, 4]
        hidden = [128, 256]
        norms = ["BatchNorm"]
        activations = ["relu", "gelu"]
        aggrs = ["mean", "add"]
        learning_rates = [1e-3, 5e-4]
        weight_decays = [1e-4, 1e-5]
        dropouts = [0.1, 0.2]
        edge_features = [True, False]
        Ks = [2, 3]
        combos = []
        for _ in range(max_combinations):
            m = random.choices(models, weights=probs, k=1)[0]
            cfg = {
                "model_name": m,
                "graph_layers": random.choice(layers),
                "graph_hidden_channels": random.choice(hidden),
                "graph_norm": random.choice(norms),
                "activation": random.choice(activations),
                "heads": 1,
                "aggr": random.choice(aggrs),
                "residual": True,
                "use_edge_features": random.choice(edge_features),
                "learning_rate": random.choice(learning_rates),
                "weight_decay": random.choice(weight_decays),
                "dropout": random.choice(dropouts)
            }
            if m in ["Cheb", "SGC"]:
                cfg["K"] = random.choice(Ks)
            combos.append(cfg)
        return combos

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
            if mol.GetNumBonds() > 0:
                bond_features = [enhanced_bond_features(b) for b in mol.GetBonds()]
                ea = torch.tensor(np.array(bond_features), dtype=torch.float32)
                if g.edge_index.size(1) == ea.size(0) * 2:
                    g.edge_attr = torch.cat([ea, ea], dim=0)
                else:
                    g.edge_attr = ea
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

def build_tdc_loaders_enhanced(
        dataset_name: str,
        batch_train: int = 128,
        batch_eval: int = 256,
        split_type: str = "scaffold",
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    ADME, from_smiles_fn = _import_tdc_and_from_smiles()
    data_api = ADME(name=dataset_name)
    try:
        split = data_api.get_split(method=split_type)
    except Exception:
        split = data_api.get_split()

    print(f"\n  DEBUG INFO   {dataset_name}:")
    print(f"Train  : {len(split['train'])}")
    print(f"Test  : {len(split['test'])}")

    y_train_arr = split['train']['Y'].values.astype(float)
    print(f"Y    : {y_train_arr.min():.3f}   {y_train_arr.max():.3f}")
    print(f"Y    : {y_train_arr.mean():.3f}")

    set_target_scaler(dataset_name, y_train_arr, force_mode="log")
    print(f"  Target scaling mode: {TARGET_SCALER['mode']} (mu={TARGET_SCALER['mu']:.3f}, sigma={TARGET_SCALER['sigma']:.3f})")

    train_list = df_to_graph_list_enhanced(from_smiles_fn, split["train"])
    valid_df = split.get("valid") if "valid" in split else split.get("val")
    valid_list = df_to_graph_list_enhanced(from_smiles_fn, valid_df) if valid_df is not None else []
    test_list = df_to_graph_list_enhanced(from_smiles_fn, split["test"])

    if not valid_list:
        n = len(train_list); k = max(1, int(0.1 * n))
        valid_list = train_list[:k]; train_list = train_list[k:]

    print(f"   : train={len(train_list)}, val={len(valid_list)}, test={len(test_list)}")

    _apply_target_transform_inplace([train_list, valid_list, test_list])
    _fit_adme_scaler(train_list)
    _standardize_adme_inplace([train_list, valid_list, test_list])

    adme_dim = int(train_list[0].adme_features.numel()) if len(train_list) else 30

    train_loader = DataLoader(train_list, batch_size=batch_train, shuffle=True)
    val_loader = DataLoader(valid_list, batch_size=batch_eval, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_eval, shuffle=False)

    # Baseline RF (log-space training, original-space report)
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error
        print(f"    Random Forest baseline...")
        X_train = np.array([data.adme_features.numpy() for data in train_list])
        X_test = np.array([data.adme_features.numpy() for data in test_list])
        y_train_log = np.log(np.maximum(0.01, np.array([d.original_y for d in train_list], dtype=float)))
        y_test_original = np.array([d.original_y for d in test_list], dtype=float)
        rf = RandomForestRegressor(n_estimators=200, random_state=RNG_SEED)
        rf.fit(X_train, y_train_log)
        pred_log = rf.predict(X_test)
        pred_original = np.exp(pred_log)
        rmse = np.sqrt(mean_squared_error(y_test_original, pred_original))
        ybar = float(y_test_original.mean())
        ss_res = np.sum((y_test_original - pred_original) ** 2)
        ss_tot = np.sum((y_test_original - ybar) ** 2) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        print(f"  Random Forest Baseline: RMSE={rmse:.3f}, R ={r2:.3f}")
    except ImportError:
        print("  sklearn       - baseline test  ")
    except Exception as e:
        print(f"  Baseline test failed: {e}")

    return train_loader, val_loader, test_loader, adme_dim

# ===================== TRAINING FUNCTIONS =====================
def train_one_epoch_enhanced(model, loader, optimizer, device, loss_fn=None, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0; num_batches = 0
    if loss_fn is None:
        loss_fn = RobustLoss(alpha=0.7)
    for data in loader:
        data = data.to(device)
        pred = model(data)
        y = data.y.view_as(pred)
        loss = loss_fn(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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

def build_model_from_config_enhanced(config: Dict, adme_dim: int, device: str):
    config_copy = config.copy()
    input_dim = config_copy.pop("input_dim", 37)
    backbone = GraphBackbone(input_dim=input_dim, **config_copy).to(device)
    model = ImprovedMolecularRegressor(backbone, adme_dim=adme_dim).to(device)
    return model

def train_eval_single_enhanced(
        config: Dict,
        dataset_name: str,
        epochs: int = 120,
        patience: int = 25,
        batch_train: int = 32,
        batch_eval: int = 64,
        device: str = None,
        seed: int = 42,
        wd_default: float = 1e-4,
) -> Tuple[Dict, Dict[str, float], Dict[str, float], Dict]:
    set_global_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, adme_dim = build_tdc_loaders_enhanced(
        dataset_name, batch_train, batch_eval, split_type="scaffold"
    )
    in_dim = int(train_loader.dataset[0].x.size(1))
    cfg = {**config, "input_dim": in_dim}
    model = build_model_from_config_enhanced(cfg, adme_dim, device)

    lr = float(config.get("learning_rate", 1e-3))
    wd = float(config.get("weight_decay", wd_default))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    loss_fn = RobustLoss(alpha=0.6 if "Hepatocyte" in dataset_name else 0.7)

    best_log_rmse = float("inf")
    best_state = None
    best_epoch = 0
    no_imp = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_enhanced(model, train_loader, optimizer, device, loss_fn, max_grad_norm=1.0)
        val_raw, val_pred_raw, val_true_raw = evaluate(model, val_loader, device, inverse=True, return_arrays=True)
        val_log_metrics = compute_metrics_np(val_true_raw, val_pred_raw, log_space=True)

        improved = val_log_metrics["rmse"] + 1e-6 < best_log_rmse
        if improved:
            best_log_rmse = val_log_metrics["rmse"]
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            no_imp = 0
        else:
            no_imp += 1

        scheduler.step(val_log_metrics["rmse"])

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch}: train_loss={train_loss:.4f}, valRMSE={val_raw['rmse']:.3f}, valMAE={val_raw['mae']:.3f}, val_logRMSE={val_log_metrics['rmse']:.3f}")

        if no_imp >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval (original scale + log-space metrics) + Spearman
    val_raw, val_pred_raw, val_true_raw = evaluate(model, val_loader, device, inverse=True, return_arrays=True)
    val_log = compute_metrics_np(val_true_raw, val_pred_raw, log_space=True)
    val_spear = spearman_metric(val_true_raw, val_pred_raw)

    test_raw, test_pred_raw, test_true_raw = evaluate(model, test_loader, device, inverse=True, return_arrays=True)
    test_log = compute_metrics_np(test_true_raw, test_pred_raw, log_space=True)
    test_spear = spearman_metric(test_true_raw, test_pred_raw)

    print(
        f"   -> valRMSE={val_raw['rmse']:.3f}, testRMSE={test_raw['rmse']:.3f}, R ={test_raw['r2']:.3f} "
        f"(val_logRMSE={val_log['rmse']:.3f}, test_logRMSE={test_log['rmse']:.3f}), "
        f"Spearman[val]={val_spear:.3f}, Spearman[test]={test_spear:.3f}"
    )

    meta = {
        **cfg,
        "dataset": dataset_name,
        "epochs_trained": int(best_epoch),
        "timestamp": datetime.now().isoformat(),
        "train_time_s": round(time.time() - t0, 2),
        "success": True,
        "seed": int(seed),
    }

    val_metrics_out = {
        "rmse": val_raw["rmse"], "mae": val_raw["mae"], "r2": val_raw["r2"],
        "log_rmse": val_log["rmse"], "spearman": float(val_spear)
    }
    test_metrics_out = {
        "rmse": test_raw["rmse"], "mae": test_raw["mae"], "r2": test_raw["r2"],
        "log_rmse": test_log["rmse"], "spearman": float(test_spear)
    }

    return meta, val_metrics_out, test_metrics_out, {"adme_dim": adme_dim, "input_dim": in_dim}

# ===================== MAIN BENCHMARK FUNCTION =====================
def run_enhanced_molecular_benchmark(
        max_combos_per_dataset: int = 20,
        epochs: int = 120,
        patience: int = 25,
        device: str = None,
        seed: int = 42,
        out_prefix: str = "fixed_molecular_gnn",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(seed)
    tester = EnhancedGNNTester()
    combos = tester.create_enhanced_parameter_combinations(max_combinations=max_combos_per_dataset)
    datasets = ["Half_Life_Obach", "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ"]
    all_rows = []

    print(f"\n    MOLECULAR GNN (Graph + ADME Features) on {device}")
    print(f"Datasets: {datasets}")
    print(f"Combos per dataset: {len(combos)}")
    print(f"Training: {epochs} epochs, patience={patience}")
    print("   :   LOG-scaling    , ADME  ,   inverse.")

    for dname in datasets:
        print(f"\n====================== Dataset: {dname} ======================")
        rows = []
        for i, cfg in enumerate(combos, 1):
            try:
                print(f"({i}/{len(combos)}) {cfg['model_name']} L={cfg['graph_layers']} H={cfg['graph_hidden_channels']} "
                      f"Edge={cfg.get('use_edge_features', False)} LR={cfg['learning_rate']}")
                meta, val_m, test_m, aux = train_eval_single_enhanced(
                    cfg, dname, epochs=epochs, patience=patience,
                    batch_train=32, batch_eval=64, device=device, seed=seed,
                )
                row = {
                    **cfg,
                    "dataset": dname,
                    "val_rmse": val_m["rmse"],
                    "val_log_rmse": val_m.get("log_rmse", np.nan),
                    "val_spearman": val_m.get("spearman", np.nan),
                    "test_rmse": test_m["rmse"],
                    "test_log_rmse": test_m.get("log_rmse", np.nan),
                    "test_spearman": test_m.get("spearman", np.nan),
                    "val_mae": val_m["mae"],
                    "test_mae": test_m["mae"],
                    "val_r2": val_m["r2"],
                    "test_r2": test_m["r2"],
                    "epochs_trained": meta["epochs_trained"],
                    "train_time_s": meta["train_time_s"],
                    "seed": seed,
                    "success": True,
                }
                rows.append(row)
                all_rows.append(row)
            except Exception as e:
                print(f"   !! Failed: {e}")
                rows.append({**cfg, "dataset": dname, "success": False, "error": str(e),
                             "timestamp": datetime.now().isoformat()})

        df = pd.DataFrame(rows)
        csv_path = f"{out_prefix}_{dname}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        ok = df[df["success"] == True].copy()
        if len(ok) > 0:
            # Prefer logRMSE if present; ties break by RMSE
            sort_cols = ["val_log_rmse", "val_rmse"] if ok["val_log_rmse"].notna().any() else ["val_rmse"]
            ok = ok.sort_values(sort_cols, ascending=[True] * len(sort_cols))
            top = ok.head(5)
            print("\n  Top-5 by val (log)RMSE:")
            for _, r in top.iterrows():
                print(f"   {r['model_name']:11} L={int(r['graph_layers'])} H={int(r['graph_hidden_channels'])} "
                      f"Edge={r.get('use_edge_features', False)} | "
                      f"val {r['val_rmse']:.3f} (log {r['val_log_rmse']:.3f}) | "
                      f"test {r['test_rmse']:.3f} (log {r['test_log_rmse']:.3f}) | "
                      f"R  {r['test_r2']:.3f} |  (test) {r['test_spearman']:.3f}")

    all_df = pd.DataFrame(all_rows)
    all_csv = f"{out_prefix}_ALL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    all_df.to_csv(all_csv, index=False)
    print(f"\n  Global results saved: {all_csv}")
    return all_df

# ===================== MAIN EXECUTION =====================
if __name__ == "__main__":
    MAX_COMBOS_PER_DATASET = 20
    EPOCHS = 120
    PATIENCE = 25
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    results = run_enhanced_molecular_benchmark(
        max_combos_per_dataset=MAX_COMBOS_PER_DATASET,
        epochs=EPOCHS,
        patience=PATIENCE,
        device=DEVICE,
        seed=RNG_SEED,
        out_prefix="fixed_molecular_gnn",
    )

    print("\n  Molecular GNN Benchmark Finished!")
    print("     :")
    print("       LOG-  +   inverse  ")
    print("     ADME   (train-only stats) + BN   [graph|ADME]")
    print("     Spearman   (val/test)   print + CSV")
    print("       search space          ")
    print("     ReduceLROnPlateau    ac  logRMSE    ")
