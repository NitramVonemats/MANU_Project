"""
PRIMER V3 - PHASE 1: Quick Wins
Adds to V2:
1. Consistency Regularization (reduce overfitting on small datasets)
2. Multimodal Fusion (GNN + SMILES Transformer / ChemBERTa)

Expected gains: +15-30% RMSE reduction over baseline
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

# ===================== NEW: Import transformers for SMILES =====================
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("WARNING: transformers not installed. Install: pip install transformers")
    print("Multimodal fusion will be disabled.")
    TRANSFORMERS_AVAILABLE = False

# ===================== RDKit SETUP =====================
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKit_OK = True
except Exception:
    RDKit_OK = False
    print("WARNING: RDKit not available")

# ===================== GLOBAL SCALERS =====================
TARGET_SCALER = {"mode": "log", "mu": 0.0, "sigma": 1.0}
ADME_SCALER = {"mu": None, "sigma": None}
EDGE_SCALER = {"mu": None, "sigma": None}

RNG_SEED = 42

# ===================== UTILS (from primer_v2.py) =====================
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

# ===================== MOLECULAR FEATURES (from primer_v2.py) =====================
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

# ===================== NEW: SMILES TOKENIZER & EMBEDDER =====================
class SMILESEncoder(nn.Module):
    """
    Encodes SMILES strings using a pre-trained transformer (ChemBERTa)
    Falls back to simple character-level embedding if transformers unavailable
    """
    def __init__(self, output_dim=128, use_pretrained=True):
        super().__init__()
        self.output_dim = output_dim
        self.use_pretrained = use_pretrained and TRANSFORMERS_AVAILABLE

        if self.use_pretrained:
            try:
                # Try to load ChemBERTa or ChemBERT
                model_name = "seyonec/ChemBERTa-zinc-base-v1"  # Pre-trained on ZINC
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.transformer = AutoModel.from_pretrained(model_name)

                # Project transformer output to desired dim
                transformer_dim = self.transformer.config.hidden_size
                self.proj = nn.Linear(transformer_dim, output_dim)

                print(f"Loaded pre-trained SMILES encoder: {model_name}")
            except Exception as e:
                print(f"Failed to load pre-trained model: {e}")
                print("Falling back to character-level encoding")
                self.use_pretrained = False

        if not self.use_pretrained:
            # Simple character-level fallback
            self.char_embedding = nn.Embedding(100, 64)  # 100 possible SMILES chars
            self.lstm = nn.LSTM(64, output_dim // 2, 2, batch_first=True, bidirectional=True)

    def forward(self, smiles_list):
        """
        Args:
            smiles_list: List of SMILES strings
        Returns:
            Tensor of shape [batch_size, output_dim]
        """
        if self.use_pretrained:
            # Tokenize SMILES
            inputs = self.tokenizer(smiles_list, padding=True, truncation=True,
                                   return_tensors="pt", max_length=512)
            inputs = {k: v.to(next(self.transformer.parameters()).device) for k, v in inputs.items()}

            # Get transformer embeddings
            outputs = self.transformer(**inputs)
            # Use [CLS] token representation
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_dim]
            return self.proj(cls_emb)
        else:
            # Character-level encoding (fallback)
            # Convert SMILES to indices
            max_len = max(len(s) for s in smiles_list)
            batch_indices = []
            for smiles in smiles_list:
                indices = [min(ord(c), 99) for c in smiles[:max_len]]
                indices += [0] * (max_len - len(indices))  # Pad
                batch_indices.append(indices)

            indices_tensor = torch.tensor(batch_indices, dtype=torch.long)
            indices_tensor = indices_tensor.to(next(self.char_embedding.parameters()).device)

            # Embed and encode
            embedded = self.char_embedding(indices_tensor)  # [batch, max_len, 64]
            _, (hidden, _) = self.lstm(embedded)  # hidden: [4, batch, output_dim//2]
            # Concatenate final forward and backward states
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch, output_dim]
            return final_hidden

# ===================== NEW: MULTIMODAL FUSION =====================
class GatedFusion(nn.Module):
    """
    Fuses graph embeddings with SMILES embeddings using learned gating
    """
    def __init__(self, graph_dim, smiles_dim, output_dim=None):
        super().__init__()
        self.graph_dim = graph_dim
        self.smiles_dim = smiles_dim
        self.output_dim = output_dim or (graph_dim + smiles_dim)

        # Gate network: learns how much to weight each modality
        self.gate = nn.Sequential(
            nn.Linear(graph_dim + smiles_dim, (graph_dim + smiles_dim) // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear((graph_dim + smiles_dim) // 2, 2),  # 2 gates: one for graph, one for SMILES
            nn.Softmax(dim=1)
        )

        # Optional projection to output_dim
        if self.output_dim != (graph_dim + smiles_dim):
            self.proj = nn.Linear(graph_dim + smiles_dim, self.output_dim)
        else:
            self.proj = None

    def forward(self, graph_emb, smiles_emb):
        """
        Args:
            graph_emb: [batch, graph_dim]
            smiles_emb: [batch, smiles_dim]
        Returns:
            Fused embedding [batch, output_dim]
        """
        # Concatenate
        concat = torch.cat([graph_emb, smiles_emb], dim=1)  # [batch, graph_dim + smiles_dim]

        # Learn gates
        gates = self.gate(concat)  # [batch, 2]
        graph_gate = gates[:, 0:1]  # [batch, 1]
        smiles_gate = gates[:, 1:2]  # [batch, 1]

        # Apply gates (soft weighting)
        graph_weighted = graph_emb * graph_gate.expand_as(graph_emb)
        smiles_weighted = smiles_emb * smiles_gate.expand_as(smiles_emb)

        # Concatenate weighted
        fused = torch.cat([graph_weighted, smiles_weighted], dim=1)

        # Project if needed
        if self.proj is not None:
            fused = self.proj(fused)

        return fused

# ===================== NEW: CONSISTENCY REGULARIZATION =====================
def graph_dropout_augmentation(data: Data, edge_drop_prob=0.1, feat_noise_std=0.05):
    """
    Augment graph by:
    1. Randomly dropping edges
    2. Adding small noise to node features
    """
    augmented = data.clone()

    # Edge dropout
    if edge_drop_prob > 0 and augmented.edge_index.size(1) > 0:
        edge_mask = torch.rand(augmented.edge_index.size(1)) > edge_drop_prob
        augmented.edge_index = augmented.edge_index[:, edge_mask]
        if hasattr(augmented, 'edge_attr') and augmented.edge_attr is not None:
            augmented.edge_attr = augmented.edge_attr[edge_mask]

    # Feature noise
    if feat_noise_std > 0:
        noise = torch.randn_like(augmented.x) * feat_noise_std
        augmented.x = augmented.x + noise

    return augmented

def consistency_loss(model, data, edge_drop_prob=0.1, feat_noise_std=0.05):
    """
    Compute consistency loss between original and augmented predictions
    """
    # Original prediction
    pred1 = model(data)

    # Augmented prediction
    data_aug = graph_dropout_augmentation(data, edge_drop_prob, feat_noise_std)
    pred2 = model(data_aug)

    # MSE between predictions (detach one to prevent gradient flow)
    return F.mse_loss(pred1, pred2.detach())

# ===================== FROM primer_v2.py: EMA, Warmup, etc. =====================
# (Copy EMA and WarmupScheduler from primer_v2.py - abbreviated here for space)
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, base_scheduler=None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.current_epoch = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, metrics=None):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr_scale = self.current_epoch / self.warmup_epochs
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.base_lrs[i] * lr_scale
        else:
            if self.base_scheduler is not None:
                if metrics is not None:
                    self.base_scheduler.step(metrics)
                else:
                    self.base_scheduler.step()

# ===================== ARCHITECTURE (adapted from primer_v2.py) =====================
# (Include GraphBackbone, GatedAttentionReadout, etc. from primer_v2 - abbreviated)

class GraphBackbone(nn.Module):
    # ... (copy from primer_v2.py)
    pass

class GatedAttentionReadout(nn.Module):
    # ... (copy from primer_v2.py)
    pass

class RobustLoss(nn.Module):
    # ... (copy from primer_v2.py)
    pass

# Continue in next message due to length...
