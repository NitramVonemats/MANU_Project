"""
PHASE 2: FOUNDATION MODEL BENCHMARKING
=======================================
Compare molecular representations for ADME prediction:
1. GNN-only (Phase 1 baseline)
2. ChemBERTa-only (pretrained transformer)
3. GNN + ChemBERTa (hybrid fusion)

Goal: Identify which foundation model is best for ADME tasks
"""

import time
from copy import deepcopy
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, GINEConv, global_mean_pool, global_max_pool

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    RDKit_OK = True
except:
    RDKit_OK = False

# Transformers (ChemBERTa)
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_OK = True
except:
    TRANSFORMERS_OK = False
    print("WARNING: transformers not available - ChemBERTa will be disabled")

# ===================== CONFIG =====================
TARGET_SCALER = {"mode": "log", "mu": 0.0, "sigma": 1.0}
ADME_SCALER = {"mu": None, "sigma": None}

# Best configs from Phase 1
PHASE1_BEST_CONFIGS = {
    "Half_Life_Obach": {
        "model_type": "SAGE",  # SAGEConv worked best
        "layers": 3,
        "hidden": 64,
        "dropout": 0.5,
        "weight_decay": 1e-3,
        "lr": 1e-3,
        "split_type": "scaffold",
    },
    "Clearance_Hepatocyte_AZ": {
        "model_type": "GINE",  # GINEConv for bond awareness
        "layers": 3,
        "hidden": 64,
        "dropout": 0.55,
        "weight_decay": 3e-3,
        "lr": 5e-4,
        "split_type": "random",
    },
    "Clearance_Microsome_AZ": {
        "model_type": "GINE",
        "layers": 3,
        "hidden": 64,
        "dropout": 0.55,
        "weight_decay": 3e-3,
        "lr": 5e-4,
        "split_type": "random",
    }
}

# ===================== UTILS =====================
try:
    from scipy.stats import spearmanr
    def spearman_metric(y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        if len(y_true) < 3:
            return 0.0
        try:
            corr = spearmanr(y_true, y_pred).correlation
            return float(corr) if not np.isnan(corr) else 0.0
        except:
            return 0.0
except:
    def spearman_metric(y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        if len(y_true) < 3:
            return 0.0
        r_true = y_true.argsort().argsort().astype(float)
        r_pred = y_pred.argsort().argsort().astype(float)
        r_true = (r_true - r_true.mean()) / (r_true.std() + 1e-12)
        r_pred = (r_pred - r_pred.mean()) / (r_pred.std() + 1e-12)
        return float((r_true * r_pred).mean())

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ybar = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - ybar) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"rmse": rmse, "mae": mae, "r2": r2}

def set_target_scaler(y_train):
    z = np.log(np.clip(y_train.astype(float), 1e-3, None))
    TARGET_SCALER.update({"mu": float(z.mean()), "sigma": float(z.std() or 1.0)})

def _transform_y(y: float) -> float:
    return (np.log(max(1e-3, float(y))) - TARGET_SCALER["mu"]) / TARGET_SCALER["sigma"]

def _inverse_y(t):
    z = t * TARGET_SCALER["sigma"] + TARGET_SCALER["mu"]
    z = np.clip(z, -10, 10)
    return np.exp(z)

# ===================== FEATURES =====================
def enhanced_atom_features(atom):
    try:
        features = [
            atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
            atom.GetTotalNumHs(), atom.GetMass(),
            int(atom.GetHybridization() == Chem.HybridizationType.SP),
            int(atom.GetHybridization() == Chem.HybridizationType.SP2),
            int(atom.GetHybridization() == Chem.HybridizationType.SP3),
            int(atom.GetHybridization() == Chem.HybridizationType.SP3D),
            int(atom.GetIsAromatic()), int(atom.IsInRing()),
            int(atom.IsInRingSize(3)), int(atom.IsInRingSize(4)),
            int(atom.IsInRingSize(5)), int(atom.IsInRingSize(6)),
            int(atom.IsInRingSize(7)),
            atom.GetTotalValence(), atom.GetImplicitValence(), atom.GetExplicitValence(),
            int(atom.GetChiralTag()),
            int(atom.GetSymbol() == 'C'), int(atom.GetSymbol() == 'N'),
            int(atom.GetSymbol() == 'O'), int(atom.GetSymbol() == 'S'),
            int(atom.GetSymbol() == 'F'), int(atom.GetSymbol() == 'Cl'),
            int(atom.GetSymbol() == 'Br'),
        ]
        return np.array(features, dtype=np.float32)
    except:
        return np.zeros(27, dtype=np.float32)

def enhanced_bond_features(bond):
    try:
        bt = bond.GetBondType()
        features = [
            int(bt == Chem.BondType.SINGLE), int(bt == Chem.BondType.DOUBLE),
            int(bt == Chem.BondType.TRIPLE), int(bt == Chem.BondType.AROMATIC),
            float(bond.GetBondTypeAsDouble()),
            int(bond.GetIsAromatic()), int(bond.GetIsConjugated()), int(bond.IsInRing()),
            int(bond.GetStereo() == Chem.BondStereo.STEREONONE),
            int(bond.GetStereo() == Chem.BondStereo.STEREOZ),
            int(bond.GetStereo() == Chem.BondStereo.STEREOE),
            int(bond.GetStereo() == Chem.BondStereo.STEREOANY),
        ]
        return np.array(features, dtype=np.float32)
    except:
        return np.array([1, 0, 0, 0, 1.0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32)

def adme_specific_descriptors(smiles: str):
    if not RDKit_OK:
        return np.zeros(20, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(20, dtype=np.float32)
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        descriptors = [
            mw, logp, hbd, hba, Descriptors.NumRotatableBonds(mol),
            rdMolDescriptors.CalcTPSA(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
            int(mw > 500), int(logp > 5), int(hbd > 5), int(hba > 10),
            Descriptors.MolMR(mol), rdMolDescriptors.CalcNumAliphaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol), Descriptors.BertzCT(mol),
            Descriptors.Chi0v(mol), Descriptors.NumHeteroatoms(mol),
            rdMolDescriptors.CalcNumHeterocycles(mol), int(mw < 200), int(mw > 800),
        ]
        return np.array(descriptors[:20], dtype=np.float32)
    except:
        return np.zeros(20, dtype=np.float32)

# ===================== FOUNDATION MODEL ENCODERS =====================

class ChemBERTaEncoder(nn.Module):
    """ChemBERTa pretrained transformer for SMILES"""
    def __init__(self, model_name="DeepChem/ChemBERTa-77M-MLM", proj_dim=256):
        super().__init__()
        self.enabled = TRANSFORMERS_OK
        self.proj_dim = proj_dim
        self.name = "ChemBERTa"

        if self.enabled:
            print(f"Loading ChemBERTa from {model_name}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                h_dim = self.model.config.hidden_size
            except Exception as e:
                print(f"WARNING: Failed to load ChemBERTa: {e}")
                self.enabled = False
                h_dim = proj_dim
        else:
            h_dim = proj_dim

        self.proj = nn.Linear(h_dim, proj_dim)

    @torch.no_grad()
    def _embed_smiles(self, smiles_list, device):
        if not self.enabled or len(smiles_list) == 0:
            return torch.zeros((len(smiles_list), self.proj_dim), device=device)

        tokens = self.tokenizer(
            smiles_list, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        ).to(device)

        output = self.model(**tokens).last_hidden_state  # [B, T, H]
        emb = output.mean(dim=1)  # Mean pooling
        return emb

    def forward(self, smiles_list, device):
        emb = self._embed_smiles(smiles_list, device)
        return self.proj(emb)


class MolFormerEncoder(nn.Module):
    """MolFormer (IBM) - Large-scale pretrained transformer"""
    def __init__(self, model_name="ibm/MoLFormer-XL-both-10pct", proj_dim=256):
        super().__init__()
        self.enabled = TRANSFORMERS_OK
        self.proj_dim = proj_dim
        self.name = "MolFormer"

        if self.enabled:
            print(f"Loading MolFormer from {model_name}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, deterministic_eval=True)
                h_dim = self.model.config.hidden_size
            except Exception as e:
                print(f"WARNING: Failed to load MolFormer: {e}")
                print("Falling back to ChemBERTa architecture...")
                self.enabled = False
                h_dim = proj_dim
        else:
            h_dim = proj_dim

        self.proj = nn.Linear(h_dim, proj_dim)

    @torch.no_grad()
    def _embed_smiles(self, smiles_list, device):
        if not self.enabled or len(smiles_list) == 0:
            return torch.zeros((len(smiles_list), self.proj_dim), device=device)

        try:
            tokens = self.tokenizer(
                smiles_list, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(device)

            output = self.model(**tokens).last_hidden_state  # [B, T, H]
            emb = output.mean(dim=1)  # Mean pooling
            return emb
        except Exception as e:
            print(f"WARNING: MolFormer embedding failed: {e}")
            return torch.zeros((len(smiles_list), self.proj_dim), device=device)

    def forward(self, smiles_list, device):
        emb = self._embed_smiles(smiles_list, device)
        return self.proj(emb)


class RobertaLikeEncoder(nn.Module):
    """Generic RoBERTa-based encoder for molecular SMILES (fallback for other models)"""
    def __init__(self, model_name="seyonec/PubChem10M_SMILES_BPE_450k", proj_dim=256):
        super().__init__()
        self.enabled = TRANSFORMERS_OK
        self.proj_dim = proj_dim
        self.name = "RoBERTa-SMILES"

        if self.enabled:
            print(f"Loading RoBERTa-based model from {model_name}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                h_dim = self.model.config.hidden_size
            except Exception as e:
                print(f"WARNING: Failed to load RoBERTa model: {e}")
                self.enabled = False
                h_dim = proj_dim
        else:
            h_dim = proj_dim

        self.proj = nn.Linear(h_dim, proj_dim)

    @torch.no_grad()
    def _embed_smiles(self, smiles_list, device):
        if not self.enabled or len(smiles_list) == 0:
            return torch.zeros((len(smiles_list), self.proj_dim), device=device)

        tokens = self.tokenizer(
            smiles_list, padding=True, truncation=True,
            max_length=256, return_tensors="pt"
        ).to(device)

        output = self.model(**tokens).last_hidden_state  # [B, T, H]
        emb = output.mean(dim=1)  # Mean pooling
        return emb

    def forward(self, smiles_list, device):
        emb = self._embed_smiles(smiles_list, device)
        return self.proj(emb)


class MorganFingerprintEncoder(nn.Module):
    """Morgan Fingerprint (ECFP) baseline - not a foundation model but good baseline"""
    def __init__(self, n_bits=2048, radius=2, proj_dim=256):
        super().__init__()
        self.n_bits = n_bits
        self.radius = radius
        self.proj_dim = proj_dim
        self.enabled = RDKit_OK
        self.name = "Morgan-FP"

        # Simple MLP encoder for fingerprints
        self.encoder = nn.Sequential(
            nn.Linear(n_bits, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, proj_dim)
        )

    def _get_fingerprint(self, smiles):
        if not self.enabled:
            return np.zeros(self.n_bits, dtype=np.float32)
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(self.n_bits, dtype=np.float32)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
            return np.array(fp, dtype=np.float32)
        except:
            return np.zeros(self.n_bits, dtype=np.float32)

    def forward(self, smiles_list, device):
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        fps = np.array([self._get_fingerprint(s) for s in smiles_list])
        fps_tensor = torch.tensor(fps, dtype=torch.float32, device=device)
        return self.encoder(fps_tensor)

# ===================== MODELS =====================
class GNNBackbone(nn.Module):
    """GNN backbone - SAGE or GINE"""
    def __init__(self, model_type="SAGE", layers=3, hidden=64, dropout=0.5,
                 input_dim=27, edge_dim=12):
        super().__init__()
        self.model_type = model_type
        self.layers = layers
        self.output_dim = hidden

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(layers):
            in_dim = input_dim if i == 0 else hidden

            if model_type == "SAGE":
                conv = SAGEConv(in_dim, hidden, aggr="mean")
            elif model_type == "GINE":
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden * 2),
                    nn.BatchNorm1d(hidden * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(hidden * 2, hidden),
                )
                conv = GINEConv(mlp, edge_dim=edge_dim, train_eps=True)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(hidden))
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None) if self.model_type == "GINE" else None

        for i, (conv, norm, dropout) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            if self.model_type == "GINE" and edge_attr is not None:
                h = conv(x, edge_index, edge_attr)
            else:
                h = conv(x, edge_index)

            h = norm(h)
            h = F.relu(h)
            h = dropout(h)

            # Residual connection
            if i > 0 and i < len(self.convs) - 1 and h.shape == x.shape:
                x = x + 0.3 * h
            else:
                x = h

        return x

class GNNOnlyRegressor(nn.Module):
    """GNN-only model (Phase 1 baseline)"""
    def __init__(self, backbone, adme_dim=20):
        super().__init__()
        self.backbone = backbone
        self.adme_dim = adme_dim

        # Readout: mean + max pooling
        graph_dim = backbone.output_dim * 2
        combined_dim = graph_dim + adme_dim

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        # Graph encoding
        graph_emb = self.backbone(data)
        mean_pool = global_mean_pool(graph_emb, data.batch)
        max_pool = global_max_pool(graph_emb, data.batch)
        graph_pooled = torch.cat([mean_pool, max_pool], dim=-1)

        # ADME features
        adme = getattr(data, "adme_features", None)
        if adme is None:
            batch_size = int(data.batch.max().item() + 1)
            adme = torch.zeros(batch_size, self.adme_dim, device=data.x.device)
        elif adme.dim() == 3:
            adme = adme.squeeze(1)
        elif adme.dim() == 1:
            adme = adme.view(-1, self.adme_dim)

        combined = torch.cat([graph_pooled, adme], dim=-1)
        return self.head(combined).squeeze(-1)

class FoundationOnlyRegressor(nn.Module):
    """Foundation model only (no GNN) - works with any text encoder"""
    def __init__(self, text_encoder, adme_dim=20, text_dim=256):
        super().__init__()
        self.adme_dim = adme_dim
        self.text_encoder = text_encoder

        combined_dim = text_dim + adme_dim

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        device = data.x.device if hasattr(data, 'x') else 'cpu'

        # SMILES → Foundation Model
        smiles_list = getattr(data, "smiles", None) or []
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        if not smiles_list:
            batch_size = int(data.batch.max().item() + 1) if hasattr(data, 'batch') else 1
            smiles_list = [""] * batch_size

        text_emb = self.text_encoder(smiles_list, device=device)

        # ADME features
        adme = getattr(data, "adme_features", None)
        if adme is None:
            batch_size = len(smiles_list)
            adme = torch.zeros(batch_size, self.adme_dim, device=device)
        elif adme.dim() == 3:
            adme = adme.squeeze(1)
        elif adme.dim() == 1:
            adme = adme.view(-1, self.adme_dim)

        combined = torch.cat([text_emb, adme], dim=-1)
        return self.head(combined).squeeze(-1)

# Backward compatibility
class ChemBERTaOnlyRegressor(FoundationOnlyRegressor):
    """ChemBERTa-only model (backward compatibility)"""
    def __init__(self, adme_dim=20, text_dim=256):
        text_encoder = ChemBERTaEncoder(proj_dim=text_dim)
        super().__init__(text_encoder, adme_dim, text_dim)

class HybridRegressor(nn.Module):
    """GNN + Foundation Model fusion - works with any text encoder"""
    def __init__(self, backbone, text_encoder, adme_dim=20, text_dim=256):
        super().__init__()
        self.backbone = backbone
        self.adme_dim = adme_dim
        self.text_encoder = text_encoder

        # Readout
        graph_dim = backbone.output_dim * 2

        # Simple concatenation fusion
        combined_dim = graph_dim + text_dim + adme_dim

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        device = data.x.device

        # Graph encoding
        graph_emb = self.backbone(data)
        mean_pool = global_mean_pool(graph_emb, data.batch)
        max_pool = global_max_pool(graph_emb, data.batch)
        graph_pooled = torch.cat([mean_pool, max_pool], dim=-1)

        # SMILES → ChemBERTa
        smiles_list = getattr(data, "smiles", None) or []
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        if not smiles_list:
            smiles_list = [""] * int(data.batch.max().item() + 1)

        text_emb = self.text_encoder(smiles_list, device=device)

        # ADME features
        adme = getattr(data, "adme_features", None)
        if adme is None:
            batch_size = int(data.batch.max().item() + 1)
            adme = torch.zeros(batch_size, self.adme_dim, device=device)
        elif adme.dim() == 3:
            adme = adme.squeeze(1)
        elif adme.dim() == 1:
            adme = adme.view(-1, self.adme_dim)

        # Concatenate all
        combined = torch.cat([graph_pooled, text_emb, adme], dim=-1)
        return self.head(combined).squeeze(-1)

# ===================== DATA LOADING =====================
def _import_tdc_and_from_smiles():
    try:
        from tdc.single_pred import ADME
    except Exception as e:
        raise RuntimeError("TDC not installed") from e
    try:
        from torch_geometric.utils import from_smiles
    except:
        try:
            from torch_geometric.data import from_smiles
        except Exception as e:
            raise RuntimeError("PyG from_smiles unavailable") from e
    return ADME, from_smiles

def row_to_graph(from_smiles_fn, smiles: str, y_value: float):
    g = from_smiles_fn(smiles)
    if g is None or getattr(g, "x", None) is None:
        return None

    if RDKit_OK:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Enhanced atom features
        atom_features = [enhanced_atom_features(atom) for atom in mol.GetAtoms()]
        g.x = torch.tensor(np.array(atom_features), dtype=torch.float32)

        # Enhanced bond features
        if mol.GetNumBonds() > 0:
            bond_features = []
            for bond in mol.GetBonds():
                bond_features.append(enhanced_bond_features(bond))
                bond_features.append(enhanced_bond_features(bond))  # Reverse edge
            g.edge_attr = torch.tensor(np.array(bond_features), dtype=torch.float32)
        else:
            g.edge_attr = torch.zeros((0, 12), dtype=torch.float32)

    g.x = g.x.float()
    g.edge_index = g.edge_index.long()
    g.original_y = float(y_value)
    g.y = torch.tensor([0.0], dtype=torch.float)
    g.adme_features = torch.tensor(adme_specific_descriptors(smiles), dtype=torch.float)
    g.smiles = smiles  # Important for ChemBERTa!

    return g

def df_to_graph_list(from_smiles_fn, df):
    out = []
    for _, r in df.iterrows():
        g = row_to_graph(from_smiles_fn, r["Drug"], r["Y"])
        if g is not None:
            out.append(g)
    return out

def build_loaders(dataset_name, split_type="scaffold", batch_train=32, batch_eval=64):
    ADME, from_smiles_fn = _import_tdc_and_from_smiles()
    data_api = ADME(name=dataset_name)

    try:
        split = data_api.get_split(method=split_type)
    except:
        split = data_api.get_split()

    print(f"  Dataset: {dataset_name} ({split_type} split)")
    print(f"  Train: {len(split['train'])}, Test: {len(split['test'])}")

    y_train = split['train']['Y'].values.astype(float)
    set_target_scaler(y_train)

    train_list = df_to_graph_list(from_smiles_fn, split["train"])
    valid_df = split.get("valid") if "valid" in split else split.get("val")
    valid_list = df_to_graph_list(from_smiles_fn, valid_df) if valid_df is not None else []
    test_list = df_to_graph_list(from_smiles_fn, split["test"])

    if not valid_list:
        n = len(train_list)
        k = max(1, int(0.15 * n))
        valid_list = train_list[:k]
        train_list = train_list[k:]

    # Transform targets
    for ds in [train_list, valid_list, test_list]:
        for g in ds:
            g.y = torch.tensor([_transform_y(g.original_y)], dtype=torch.float)

    # Fit ADME scaler
    X = np.stack([d.adme_features.numpy() for d in train_list], axis=0)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    ADME_SCALER["mu"] = torch.tensor(mu, dtype=torch.float32)
    ADME_SCALER["sigma"] = torch.tensor(sigma, dtype=torch.float32)

    # Standardize ADME
    for ds in [train_list, valid_list, test_list]:
        for g in ds:
            g.adme_features = (g.adme_features - ADME_SCALER["mu"]) / (ADME_SCALER["sigma"] + 1e-8)

    train_loader = DataLoader(train_list, batch_size=batch_train, shuffle=True)
    val_loader = DataLoader(valid_list, batch_size=batch_eval, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_eval, shuffle=False)

    adme_dim = int(train_list[0].adme_features.numel())

    return train_loader, val_loader, test_loader, adme_dim

# ===================== TRAINING =====================
def train_one_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        pred = model(data)
        y = data.y.view_as(pred)
        loss = loss_fn(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, device, inverse=False):
    model.eval()
    preds, trues = [], []
    for data in loader:
        data = data.to(device)
        p = model(data)
        preds.append(p.detach().cpu())
        trues.append(data.y.view_as(p).detach().cpu())
    if not preds:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}, np.array([]), np.array([])
    preds = torch.cat(preds).float().numpy()
    trues = torch.cat(trues).float().numpy()
    if inverse:
        preds = _inverse_y(preds)
        trues = _inverse_y(trues)
    return compute_metrics_np(trues, preds), preds, trues

def train_model(model_name, dataset_name, model, config, seed=42, epochs=150, patience=30, device=None):
    """Train a single model"""
    set_global_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[{model_name}] Training on {dataset_name}...")

    train_loader, val_loader, test_loader, adme_dim = build_loaders(
        dataset_name, split_type=config['split_type'], batch_train=32, batch_eval=64
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    loss_fn = nn.MSELoss()

    best_metric = float("inf")
    best_state = None
    best_epoch = 0
    no_imp = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_metrics, _, _ = evaluate(model, val_loader, device, inverse=True)

        monitor_metric = val_metrics["rmse"]

        if monitor_metric < best_metric:
            best_metric = monitor_metric
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            no_imp = 0
        else:
            no_imp += 1

        scheduler.step(monitor_metric)

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch}: loss={train_loss:.4f}, val_R2={val_metrics['r2']:.3f}")

        if no_imp >= patience:
            print(f"    Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics, val_pred, val_true = evaluate(model, val_loader, device, inverse=True)
    test_metrics, test_pred, test_true = evaluate(model, test_loader, device, inverse=True)
    val_spear = spearman_metric(val_true, val_pred)
    test_spear = spearman_metric(test_true, test_pred)

    print(f"  [{model_name}] FINAL: test_RMSE={test_metrics['rmse']:.3f}, test_R2={test_metrics['r2']:.3f}, Spear={test_spear:.3f}")

    return {
        "model_name": model_name,
        "dataset": dataset_name,
        "seed": seed,
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_spearman": test_spear,
        "val_r2": val_metrics["r2"],
        "epochs_trained": best_epoch,
        "train_time_s": round(time.time() - t0, 2),
    }

def benchmark_dataset(dataset_name, seeds=[42, 123, 456], device=None):
    """Benchmark all foundation models on a single dataset"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = PHASE1_BEST_CONFIGS[dataset_name]

    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {dataset_name}")
    print(f"{'='*80}")

    results = []

    # Get dimensions (need to load data once)
    train_loader, _, _, adme_dim = build_loaders(dataset_name, split_type=config['split_type'])
    in_dim = int(train_loader.dataset[0].x.size(1))
    edge_dim = int(train_loader.dataset[0].edge_attr.size(1)) if train_loader.dataset[0].edge_attr.numel() > 0 else 12

    # Define foundation models to test
    foundation_models = []

    # 1. ChemBERTa
    if TRANSFORMERS_OK:
        foundation_models.append(("ChemBERTa", ChemBERTaEncoder(proj_dim=256)))

    # 2. MolFormer
    if TRANSFORMERS_OK:
        foundation_models.append(("MolFormer", MolFormerEncoder(proj_dim=256)))

    # 3. RoBERTa-SMILES
    if TRANSFORMERS_OK:
        try:
            foundation_models.append(("RoBERTa-SMILES", RobertaLikeEncoder(proj_dim=256)))
        except Exception as e:
            print(f"Warning: Could not load RoBERTa-SMILES: {e}")

    # 4. Morgan Fingerprints (baseline)
    foundation_models.append(("Morgan-FP", MorganFingerprintEncoder(proj_dim=256)))

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        # 1. GNN-only (baseline from Phase 1)
        print(f"\n[1/4] Testing GNN-only...")
        backbone = GNNBackbone(
            model_type=config['model_type'],
            layers=config['layers'],
            hidden=config['hidden'],
            dropout=config['dropout'],
            input_dim=in_dim,
            edge_dim=edge_dim
        )
        model_gnn = GNNOnlyRegressor(backbone, adme_dim=adme_dim)
        result_gnn = train_model("GNN-only", dataset_name, model_gnn, config, seed=seed, device=device)
        results.append(result_gnn)

        # 2. Foundation models only (without GNN)
        for idx, (fm_name, fm_encoder) in enumerate(foundation_models, start=2):
            print(f"\n[{idx}/4] Testing {fm_name}-only...")
            try:
                model_fm = FoundationOnlyRegressor(fm_encoder, adme_dim=adme_dim, text_dim=256)
                result_fm = train_model(f"{fm_name}-only", dataset_name, model_fm, config, seed=seed, device=device)
                results.append(result_fm)
            except Exception as e:
                print(f"ERROR: Failed to train {fm_name}-only: {e}")

        # 3. Hybrid models (GNN + Foundation)
        hybrid_idx = len(foundation_models) + 2
        for idx, (fm_name, fm_encoder) in enumerate(foundation_models, start=hybrid_idx):
            print(f"\n[{idx}/{hybrid_idx + len(foundation_models) - 1}] Testing GNN+{fm_name} Hybrid...")
            try:
                backbone_hybrid = GNNBackbone(
                    model_type=config['model_type'],
                    layers=config['layers'],
                    hidden=config['hidden'],
                    dropout=config['dropout'],
                    input_dim=in_dim,
                    edge_dim=edge_dim
                )
                model_hybrid = HybridRegressor(backbone_hybrid, fm_encoder, adme_dim=adme_dim, text_dim=256)
                result_hybrid = train_model(f"GNN+{fm_name}", dataset_name, model_hybrid, config, seed=seed, device=device)
                results.append(result_hybrid)
            except Exception as e:
                print(f"ERROR: Failed to train GNN+{fm_name} hybrid: {e}")

    return results

# ===================== MAIN =====================
if __name__ == "__main__":
    print(f"\n{'#'*80}")
    print("# PHASE 2: FOUNDATION MODEL BENCHMARKING - EXTENDED")
    print(f"{'#'*80}")
    print("\nComparing Multiple Foundation Models:")
    print("  1. GNN-only (Phase 1 baseline)")
    print("  2. ChemBERTa-only (DeepChem/ChemBERTa-77M-MLM)")
    print("  3. MolFormer-only (IBM MolFormer-XL)")
    print("  4. RoBERTa-SMILES-only (PubChem pretrained)")
    print("  5. Morgan-FP-only (Fingerprint baseline)")
    print("  6. GNN + ChemBERTa (hybrid fusion)")
    print("  7. GNN + MolFormer (hybrid fusion)")
    print("  8. GNN + RoBERTa-SMILES (hybrid fusion)")
    print("  9. GNN + Morgan-FP (hybrid fusion)")
    print(f"\nTransformers available: {TRANSFORMERS_OK}")
    print(f"RDKit available: {RDKit_OK}")
    print(f"\nThis will test ~9 model variants × 3 datasets × 3 seeds = ~81 experiments")
    print(f"Estimated time: 4-8 hours (depending on hardware)\n")

    datasets = ["Half_Life_Obach", "Clearance_Microsome_AZ", "Clearance_Hepatocyte_AZ"]
    all_results = []

    for dataset in datasets:
        results = benchmark_dataset(dataset, seeds=[42, 123, 456])
        all_results.extend(results)

    # Save results
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"tests/phase2_foundation_benchmark_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n\nResults saved to: {csv_path}")

    # Also save a summary
    summary_path = f"tests/phase2_summary_{timestamp}.csv"
    summary_data = []
    for dataset in datasets:
        df_dataset = df[df['dataset'] == dataset]
        for model_name in sorted(df_dataset['model_name'].unique()):
            df_model = df_dataset[df_dataset['model_name'] == model_name]
            summary_data.append({
                'dataset': dataset,
                'model': model_name,
                'mean_test_r2': df_model['test_r2'].mean(),
                'std_test_r2': df_model['test_r2'].std(),
                'mean_test_rmse': df_model['test_rmse'].mean(),
                'std_test_rmse': df_model['test_rmse'].std(),
                'mean_test_spearman': df_model['test_spearman'].mean(),
                'n_seeds': len(df_model)
            })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    # Summary by model type
    print(f"\n{'='*80}")
    print("SUMMARY BY MODEL TYPE:")
    print(f"{'='*80}")

    for dataset in datasets:
        print(f"\n{dataset}:")
        df_dataset = df[df['dataset'] == dataset]
        for model_name in df_dataset['model_name'].unique():
            df_model = df_dataset[df_dataset['model_name'] == model_name]
            mean_r2 = df_model['test_r2'].mean()
            std_r2 = df_model['test_r2'].std()
            mean_rmse = df_model['test_rmse'].mean()
            print(f"  {model_name:20} R2={mean_r2:6.3f} ± {std_r2:.3f}  RMSE={mean_rmse:6.2f}")

    print(f"\n{'='*80}")
    print("Phase 2 Complete!")
    print(f"{'='*80}")
