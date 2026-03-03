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
import itertools
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

GLOBAL_Y_MEDIAN = 0.0
GLOBAL_Y_MAD = 1.0


# ===================== ENHANCED MOLECULAR FEATURES =====================

def enhanced_atom_features(atom):
    """Extract comprehensive atom features for molecular graphs (37 dims)."""
    try:
        features = [
            # Basic atomic properties (8)
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            atom.GetTotalNumHs(),
            atom.GetMass(),

            # Ring membership (6)
            int(atom.IsInRingSize(3)),
            int(atom.IsInRingSize(4)),
            int(atom.IsInRingSize(5)),
            int(atom.IsInRingSize(6)),
            int(atom.IsInRingSize(7)),
            int(atom.IsInRingSize(8)),

            # Chemical environment (4)
            atom.GetTotalValence(),
            atom.GetImplicitValence(),
            atom.GetExplicitValence(),
            int(atom.GetChiralTag()),

            # Connectivity (2)
            len(atom.GetNeighbors()),
            atom.GetTotalDegree(),

            # Electronic properties (1)
            atom.GetAtomicNum() / 100.0,

            # Chemical type indicators (9)
            int(atom.GetSymbol() == 'C'),
            int(atom.GetSymbol() == 'N'),
            int(atom.GetSymbol() == 'O'),
            int(atom.GetSymbol() == 'S'),
            int(atom.GetSymbol() == 'P'),
            int(atom.GetSymbol() == 'F'),
            int(atom.GetSymbol() == 'Cl'),
            int(atom.GetSymbol() == 'Br'),
            int(atom.GetSymbol() == 'I'),

            # Hybridization states (5)
            int(atom.GetHybridization() == Chem.HybridizationType.SP),
            int(atom.GetHybridization() == Chem.HybridizationType.SP2),
            int(atom.GetHybridization() == Chem.HybridizationType.SP3),
            int(atom.GetHybridization() == Chem.HybridizationType.SP3D),
            int(atom.GetHybridization() == Chem.HybridizationType.SP3D2),

            # ADME-relevant features (2)
            int(atom.GetSymbol() in ['N', 'O']),  # donor potential
            int(atom.GetSymbol() in ['N', 'O', 'F']),  # acceptor potential
        ]
        return np.array(features, dtype=np.float32)
    except Exception:
        return np.array([
                            atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
                            int(atom.GetHybridization()), int(atom.GetIsAromatic()),
                            int(atom.IsInRing()), atom.GetTotalNumHs(), atom.GetMass()
                        ] + [0.0] * 29, dtype=np.float32)


def enhanced_bond_features(bond):
    """Extract comprehensive bond features (12 dims)."""
    try:
        features = [
            float(bond.GetBondTypeAsDouble()),
            int(bond.GetBondType() == Chem.BondType.SINGLE),
            int(bond.GetBondType() == Chem.BondType.DOUBLE),
            int(bond.GetBondType() == Chem.BondType.TRIPLE),
            int(bond.GetBondType() == Chem.BondType.AROMATIC),

            int(bond.GetIsAromatic()),
            int(bond.GetIsConjugated()),
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
    """ADME-specific molecular descriptors optimized for clearance/half-life (30 dims)."""
    if not RDKit_OK:
        return np.zeros(30, dtype=np.float32)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(30, dtype=np.float32)

    try:
        # Core ADME descriptors
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

        # Ensure exactly 30 features
        descriptors = descriptors[:30]
        while len(descriptors) < 30:
            descriptors.append(0.0)

        return np.array(descriptors, dtype=np.float32)
    except Exception:
        return np.zeros(30, dtype=np.float32)


# ===================== FIXED ARCHITECTURE =====================

class GraphBackbone(nn.Module):
    """Само graph backbone - производи node embeddings"""

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

            # Normalization
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

        # Enhanced activation
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

        for i, (conv, norm, dropout) in enumerate(zip(self._convs, self._norms, self._dropouts)):
            # Pass edge_attr only for layers that accept it
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
    """Подобрен readout механизам"""

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
        # H (mean) + H (max) + H (attn) + H/2 (local_mean) = 3.5H
        self.out_dim = (in_dim * 3) + (in_dim // 2)

    def forward(self, x, batch):
        mean_p = global_mean_pool(x, batch)
        max_p = global_max_pool(x, batch)
        att_p = self.attn(x, batch)
        local_x = self.local_pool(x)
        local_mean = global_mean_pool(local_x, batch)
        return torch.cat([mean_p, max_p, att_p, local_mean], dim=-1)


class ImprovedMolecularRegressor(nn.Module):
    """ПРАВИЛНА АРХИТЕКТУРА: Комбинира graph + ADME features"""

    def __init__(self, backbone: GraphBackbone, adme_dim: int = 30):
        super().__init__()
        self.backbone = backbone
        self.adme_dim = adme_dim

        # Multi-scale readout за graph features
        self.readout = MultiScaleReadout(backbone.output_dim)

        # Комбинирани features: graph + ADME
        combined_dim = self.readout.out_dim + adme_dim

        # Подобра архитектура со повеќе слоеви
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data: Data) -> torch.Tensor:
        # Graph features
        graph_emb = self.backbone(data)  # [N, graph_hidden_channels]
        graph_pooled = self.readout(graph_emb, data.batch)  # [B, readout.out_dim]

        # ADME features
        adme = getattr(data, "adme_features", None)
        if adme is None:
            batch_size = data.batch.max() + 1 if hasattr(data, 'batch') else 1
            adme = torch.zeros(batch_size, self.adme_dim, device=data.x.device)
        elif adme.dim() == 3 and adme.size(1) == 1:
            adme = adme.squeeze(1)
        elif adme.dim() == 1:
            adme = adme.view(-1, self.adme_dim)

        # Комбинирај graph + ADME
        combined = torch.cat([graph_pooled, adme], dim=-1)
        return self.head(combined).squeeze(-1)


class RobustLoss(nn.Module):
    """Combined loss function for better stability"""

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
        """Enhanced parameter combinations with edge features"""
        models = ["GCN", "SAGE", "GIN", "GAT", "TAG", "SGC", "Transformer", "Graph"]
        layers = [2, 3, 4]
        hidden = [64, 128, 256]
        norms = ["BatchNorm", "GraphNorm", "LayerNorm"]
        activations = ["relu", "gelu", "leaky_relu", "swish"]
        heads = [2, 4, 8]
        aggrs = ["mean", "max", "add"]
        learning_rates = [1e-4, 5e-4, 1e-3]
        weight_decays = [1e-5, 1e-4, 1e-3]
        dropouts = [0.1, 0.2, 0.3]
        edge_features = [True, False]
        Ks = [2, 3, 5]

        combos = []
        for _ in range(max_combinations):
            m = random.choice(models)
            cfg = {
                "model_name": m,
                "graph_layers": random.choice(layers),
                "graph_hidden_channels": random.choice(hidden),
                "graph_norm": random.choice(norms),
                "activation": random.choice(activations),
                "heads": random.choice(heads) if m in ["GAT", "Transformer"] else 1,
                "aggr": random.choice(aggrs),
                "residual": random.choice([True, False]),
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
    """Enhanced graph creation with edge features and ADME descriptors"""
    g = from_smiles_fn(smiles)
    if g is None or getattr(g, "x", None) is None:
        return None

    # Enhanced atom and bond features
    if RDKit_OK:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Atom features -> 37 dims
            atom_features = [enhanced_atom_features(atom) for atom in mol.GetAtoms()]
            g.x = torch.tensor(np.array(atom_features), dtype=torch.float32)

            # Bond features -> 12 dims and align with edge_index
            if mol.GetNumBonds() > 0:
                bond_features = [enhanced_bond_features(b) for b in mol.GetBonds()]
                ea = torch.tensor(np.array(bond_features), dtype=torch.float32)
                if g.edge_index.size(1) == ea.size(0) * 2:
                    g.edge_attr = torch.cat([ea, ea], dim=0)
                else:
                    g.edge_attr = ea

    g.x = g.x.float()
    g.edge_index = g.edge_index.long()

    # ЗАЧУВАЈ ОРИГИНАЛНА Y ВРЕДНОСТ
    g.original_y = float(y_value)
    g.y = torch.tensor([np.log(max(0.01, float(y_value)))], dtype=torch.float)

    # ADME-specific descriptors
    adme_desc = adme_specific_descriptors(smiles)
    g.adme_features = torch.tensor(adme_desc, dtype=torch.float)

    return g


def df_to_graph_list_enhanced(from_smiles_fn, df: pd.DataFrame) -> List[Data]:
    out: List[Data] = []
    for _, r in df.iterrows():
        smi = r["Drug"]
        yv = r["Y"]
        g = row_to_graph_enhanced(from_smiles_fn, smi, yv)
        if g is not None:
            out.append(g)
    return out


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

    print(f"\n🔍 DEBUG INFO за {dataset_name}:")
    print(f"Train размер: {len(split['train'])}")
    print(f"Test размер: {len(split['test'])}")

    train_y = split['train']['Y'].values
    print(f"Y вредности опсег: {train_y.min():.3f} до {train_y.max():.3f}")
    print(f"Y средна вредност: {train_y.mean():.3f}")

    train_list = df_to_graph_list_enhanced(from_smiles_fn, split["train"])
    valid_df = split["valid"] if "valid" in split else split.get("val", None)
    valid_list = df_to_graph_list_enhanced(from_smiles_fn, valid_df) if valid_df is not None else []
    test_list = df_to_graph_list_enhanced(from_smiles_fn, split["test"])

    if not valid_list:
        n = len(train_list)
        k = max(1, int(0.1 * n))
        valid_list = train_list[:k]
        train_list = train_list[k:]

    print(f"График листи: train={len(train_list)}, val={len(valid_list)}, test={len(test_list)}")

    # ПОДОБРА СТАНДАРДИЗАЦИЈА - LOG SCALING
    if len(train_list) > 0:
        global GLOBAL_Y_MEDIAN, GLOBAL_Y_MAD
        # Користи log трансформирани вредности
        y_values_log = [float(g.y.item()) for g in train_list]
        y_mean_log = np.mean(y_values_log)
        y_std_log = np.std(y_values_log)

        # Зачувај за inverse трансформација
        GLOBAL_Y_MEDIAN = y_mean_log
        GLOBAL_Y_MAD = y_std_log

        print(f"📊 Y статистики (log space) - mean: {y_mean_log:.3f}, std: {y_std_log:.3f}")

        # Standard scaling во log space
        for datasets in [train_list, valid_list, test_list]:
            for g in datasets:
                g.y = torch.tensor([(float(g.y.item()) - y_mean_log) / y_std_log], dtype=torch.float)

        print(f"✅ Log scaling стандардизација завршена!")

    adme_dim = int(train_list[0].adme_features.numel()) if len(train_list) else 30
    train_loader = DataLoader(train_list, batch_size=batch_train, shuffle=True)
    val_loader = DataLoader(valid_list, batch_size=batch_eval, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_eval, shuffle=False)

    # BASELINE ТЕСТ - LOG SCALING
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score

        print(f"🔍 Тестирање Random Forest baseline...")

        X_train = np.array([data.adme_features.numpy() for data in train_list])
        X_test = np.array([data.adme_features.numpy() for data in test_list])

        # Користи оригинални Y вредности за RF
        y_train_original = np.array([data.original_y for data in train_list])
        y_test_original = np.array([data.original_y for data in test_list])

        # Log трансформација за RF
        y_train_log = np.log(np.maximum(0.01, y_train_original))
        y_test_log = np.log(np.maximum(0.01, y_test_original))

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train_log)
        pred_log = rf.predict(X_test)

        # Конвертирај назад во оригинален scale
        pred_original = np.exp(pred_log)

        rmse = np.sqrt(mean_squared_error(y_test_original, pred_original))
        r2 = r2_score(y_test_original, pred_original)

        print(f"🏆 Random Forest Baseline: RMSE={rmse:.3f}, R²={r2:.3f}")

    except ImportError:
        print("⚠️ sklearn не е инсталирано - baseline test прескокнат")
    except Exception as e:
        print(f"⚠️ Baseline test failed: {e}")

    return train_loader, val_loader, test_loader, adme_dim


# ===================== TRAINING FUNCTIONS =====================

def set_seed(seed: int = 42):
    import random as pyrand
    pyrand.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch_enhanced(model, loader, optimizer, device, loss_fn=None, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    num_batches = 0

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
def evaluate(model, loader, device, inverse: bool = False) -> Dict[str, float]:
    model.eval()
    preds, trues = [], []
    for data in loader:
        data = data.to(device)
        p = model(data)
        preds.append(p.detach().cpu())
        trues.append(data.y.view_as(p).detach().cpu())

    if not preds:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    preds = torch.cat(preds)
    trues = torch.cat(trues)

    if inverse:
        global GLOBAL_Y_MEDIAN, GLOBAL_Y_MAD
        # Inverse од standardized log space во original space
        preds_log = preds * GLOBAL_Y_MAD + GLOBAL_Y_MEDIAN
        trues_log = trues * GLOBAL_Y_MAD + GLOBAL_Y_MEDIAN

        # Експоненцијација за да вратиме во оригинален scale
        preds = torch.exp(preds_log)
        trues = torch.exp(trues_log)

    mse = torch.mean((preds - trues) ** 2).item()
    mae = torch.mean(torch.abs(preds - trues)).item()
    var = torch.var(trues, unbiased=False).item()
    r2 = 1.0 - (mse / (var + 1e-12))
    return {"rmse": math.sqrt(mse), "mae": mae, "r2": r2}

def build_model_from_config_enhanced(config: Dict, adme_dim: int, device: str):
    """ПОПРАВЕНА ФУНКЦИЈА - користи правилна архитектура"""
    config_copy = config.copy()
    input_dim = config_copy.pop("input_dim", 37)

    backbone = GraphBackbone(input_dim=input_dim, **config_copy).to(device)
    model = ImprovedMolecularRegressor(backbone, adme_dim=adme_dim).to(device)
    return model


def train_eval_single_enhanced(
        config: Dict,
        dataset_name: str,
        epochs: int = 100,
        patience: int = 20,
        batch_train: int = 32,
        batch_eval: int = 64,
        device: str = None,
        seed: int = 42,
        wd_default: float = 1e-4,
) -> Tuple[Dict, Dict[str, float], Dict[str, float], Dict]:
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, adme_dim = build_tdc_loaders_enhanced(
        dataset_name, batch_train, batch_eval, split_type="scaffold"
    )

    # Dynamically infer input_dim from data
    in_dim = int(train_loader.dataset[0].x.size(1))
    cfg = {**config, "input_dim": in_dim}

    model = build_model_from_config_enhanced(cfg, adme_dim, device)

    lr = float(config.get("learning_rate", 1e-3))
    wd = float(config.get("weight_decay", wd_default))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    loss_fn = RobustLoss(alpha=0.7)
    best = {"val_rmse": float("inf"), "state": None, "epoch": 0}
    no_imp = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_enhanced(model, train_loader, optimizer, device, loss_fn, max_grad_norm=1.0)
        val_metrics = evaluate(model, val_loader, device, inverse=True)
        scheduler.step()

        if val_metrics["rmse"] + 1e-6 < best["val_rmse"]:
            best.update(val_rmse=val_metrics["rmse"], state=deepcopy(model.state_dict()), epoch=epoch)
            no_imp = 0
        else:
            no_imp += 1

        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: train_loss={train_loss:.4f}, val_rmse={val_metrics['rmse']:.3f}")

        if no_imp >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    val_metrics = evaluate(model, val_loader, device, inverse=True)
    test_metrics = evaluate(model, test_loader, device, inverse=True)

    meta = {
        **cfg,
        "dataset": dataset_name,
        "epochs_trained": int(best["epoch"]),
        "timestamp": datetime.now().isoformat(),
        "train_time_s": round(time.time() - t0, 2),
        "success": True,
        "seed": int(seed),
    }
    return meta, val_metrics, test_metrics, {"adme_dim": adme_dim, "input_dim": in_dim}


# ===================== MAIN BENCHMARK FUNCTION =====================

def run_enhanced_molecular_benchmark(
        max_combos_per_dataset: int = 20,
        epochs: int = 100,
        patience: int = 20,
        device: str = None,
        seed: int = 42,
        out_prefix: str = "fixed_molecular_gnn",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tester = EnhancedGNNTester()
    combos = tester.create_enhanced_parameter_combinations(max_combinations=max_combos_per_dataset)
    datasets = ["Half_Life_Obach", "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ"]
    all_rows = []

    print(f"\n🧬 ПОПРАВЕН MOLECULAR GNN (Graph + ADME Features) on {device}")
    print(f"Datasets: {datasets}")
    print(f"Combos per dataset: {len(combos)}")
    print(f"Training: {epochs} epochs, patience={patience}\n")

    for dname in datasets:
        print(f"\n====================== Dataset: {dname} ======================")
        rows = []

        for i, cfg in enumerate(combos, 1):
            try:
                print(
                    f"({i}/{len(combos)}) {cfg['model_name']} L={cfg['graph_layers']} H={cfg['graph_hidden_channels']} "
                    f"Edge={cfg.get('use_edge_features', False)} LR={cfg['learning_rate']}")

                meta, val_m, test_m, aux = train_eval_single_enhanced(
                    cfg,
                    dname,
                    epochs=epochs,
                    patience=patience,
                    batch_train=32,
                    batch_eval=64,
                    device=device,
                    seed=seed,
                )

                row = {
                    **cfg,
                    "dataset": dname,
                    "val_rmse": val_m["rmse"],
                    "test_rmse": test_m["rmse"],
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
                print(f"   -> valRMSE={val_m['rmse']:.3f}, testRMSE={test_m['rmse']:.3f}, R²={test_m['r2']:.3f}")

            except Exception as e:
                print(f"   !! Failed: {e}")
                rows.append({**cfg, "dataset": dname, "success": False, "error": str(e),
                             "timestamp": datetime.now().isoformat()})

        df = pd.DataFrame(rows)
        csv_path = f"{out_prefix}_{dname}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved: {csv_path}")

        # Прикажи најдобри резултати
        ok = df[df["success"] == True]
        if len(ok) > 0:
            top = ok.nsmallest(5, "val_rmse")
            print("\n🏆 Top-5 by val RMSE:")
            for _, r in top.iterrows():
                print(f"   {r['model_name']:11} L={int(r['graph_layers'])} H={int(r['graph_hidden_channels'])} "
                      f"Edge={r.get('use_edge_features', False)} | val {r['val_rmse']:.3f} | "
                      f"test {r['test_rmse']:.3f} | R² {r['test_r2']:.3f}")

    all_df = pd.DataFrame(all_rows)
    all_csv = f"{out_prefix}_ALL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    all_df.to_csv(all_csv, index=False)
    print(f"\n📦 Global results saved: {all_csv}")
    return all_df


# ===================== MAIN EXECUTION =====================

if __name__ == "__main__":
    MAX_COMBOS_PER_DATASET = 20
    EPOCHS = 100
    PATIENCE = 20
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    results = run_enhanced_molecular_benchmark(
        max_combos_per_dataset=MAX_COMBOS_PER_DATASET,
        epochs=EPOCHS,
        patience=PATIENCE,
        device=DEVICE,
        seed=42,
        out_prefix="fixed_molecular_gnn",
    )

    print("\n✅ Поправен Molecular GNN Benchmark Complete!")
    print("🔬 Клучни подобрувања:")
    print("   • Комбинирани graph + ADME features")
    print("   • Подобра архитектура со повеќе слоеви")
    print("   • Robust scaling наместо log трансформација")
    print("   • Подобрен training loop")
    print("   • Повеќе хиперпараметарски комбинации")