"""
Full prediction models: GNN-only, Foundation-only, and Hybrid architectures
"""
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool


class GNNOnlyRegressor(nn.Module):
    """
    GNN-only regression model (Phase 1 baseline)

    Args:
        backbone: GNN backbone module
        adme_dim: ADME descriptor dimension
    """

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
        """Forward pass for GNN-only model"""
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
    """
    Foundation model only (no GNN) - works with any text encoder

    Args:
        text_encoder: Foundation model encoder (ChemBERTa, MolFormer, etc.)
        adme_dim: ADME descriptor dimension
        text_dim: Text encoder output dimension
    """

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
        """Forward pass for foundation-only model"""
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


class HybridRegressor(nn.Module):
    """
    GNN + Foundation Model fusion - works with any text encoder

    Args:
        backbone: GNN backbone module
        text_encoder: Foundation model encoder
        adme_dim: ADME descriptor dimension
        text_dim: Text encoder output dimension
    """

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
        """Forward pass for hybrid model"""
        device = data.x.device

        # Graph encoding
        graph_emb = self.backbone(data)
        mean_pool = global_mean_pool(graph_emb, data.batch)
        max_pool = global_max_pool(graph_emb, data.batch)
        graph_pooled = torch.cat([mean_pool, max_pool], dim=-1)

        # SMILES → Foundation Model
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
