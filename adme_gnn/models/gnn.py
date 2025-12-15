"""
Graph Neural Network backbone architectures
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GINEConv, GCNConv, GATConv, GINConv


class GNNBackbone(nn.Module):
    """
    GNN backbone supporting multiple architectures

    Args:
        model_type: "GCN", "GAT", "SAGE", "GIN", or "GINE"
        layers: Number of GNN layers
        hidden: Hidden dimension
        dropout: Dropout rate
        input_dim: Input node feature dimension
        edge_dim: Edge feature dimension (for GINE)
        heads: Number of attention heads (for GAT)
    """

    def __init__(self, model_type="SAGE", layers=3, hidden=64, dropout=0.5,
                 input_dim=27, edge_dim=12, heads=4):
        super().__init__()
        self.model_type = model_type
        self.layers = layers
        self.heads = heads

        # For GAT, output dim is hidden * heads (except last layer)
        if model_type == "GAT":
            self.output_dim = hidden
        else:
            self.output_dim = hidden

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(layers):
            in_dim = input_dim if i == 0 else hidden

            # For GAT, input to middle layers is hidden * heads
            if model_type == "GAT" and i > 0:
                in_dim = hidden * heads

            if model_type == "GCN":
                conv = GCNConv(in_dim, hidden)

            elif model_type == "GAT":
                # Last layer uses single head for simplicity
                out_heads = 1 if i == layers - 1 else heads
                conv = GATConv(in_dim, hidden, heads=out_heads, dropout=dropout)

            elif model_type == "SAGE":
                conv = SAGEConv(in_dim, hidden, aggr="mean")

            elif model_type == "GIN":
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden * 2),
                    nn.BatchNorm1d(hidden * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(hidden * 2, hidden),
                )
                conv = GINConv(mlp, train_eps=True)

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

            # Norm output dimension depends on GAT heads
            norm_dim = hidden * heads if model_type == "GAT" and i < layers - 1 else hidden
            self.norms.append(nn.BatchNorm1d(norm_dim))
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, data):
        """
        Forward pass through GNN layers

        Args:
            data: PyG Data batch

        Returns:
            Node embeddings
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)

        for i, (conv, norm, dropout) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            # Apply convolution based on model type
            if self.model_type == "GINE" and edge_attr is not None:
                h = conv(x, edge_index, edge_attr)
            else:
                h = conv(x, edge_index)

            h = norm(h)
            h = F.relu(h)
            h = dropout(h)

            # Residual connection (skip middle layers, if dimensions match)
            if i > 0 and i < len(self.convs) - 1 and h.shape == x.shape:
                x = x + 0.3 * h
            else:
                x = h

        return x
