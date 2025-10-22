import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GINConv, GATConv,
    TAGConv, ChebConv, SGConv, TransformerConv,
    GraphConv, global_mean_pool, LayerNorm, GraphNorm
)
import itertools
import random


class UniversalGraphStack(nn.Module):
    def __init__(self,
                 model_name="GCN",
                 graph_layers=3,
                 graph_hidden_channels=64,
                 graph_norm="BatchNorm",
                 activation="relu",
                 heads=1,
                 aggr="mean",
                 residual=True,
                 **kwargs):
        super().__init__()
        self.model_name = model_name
        self.graph_layers = graph_layers
        self.output_dim = graph_hidden_channels
        self.residual = residual

        convs, norms = [], []
        in_dim = kwargs.get("input_dim", 38)
        for i in range(graph_layers):
            if model_name == "GCN":
                conv = GCNConv(in_dim if i == 0 else graph_hidden_channels, graph_hidden_channels)
            elif model_name == "SAGE":
                conv = SAGEConv(in_dim if i == 0 else graph_hidden_channels, graph_hidden_channels, aggr=aggr)
            elif model_name == "GIN":
                nn_lin = nn.Sequential(nn.Linear(in_dim if i == 0 else graph_hidden_channels, graph_hidden_channels),
                                       nn.ReLU(),
                                       nn.Linear(graph_hidden_channels, graph_hidden_channels))
                conv = GINConv(nn_lin)
            elif model_name == "GAT":
                conv = GATConv(in_dim if i == 0 else graph_hidden_channels,
                               graph_hidden_channels // heads,
                               heads=heads)
            elif model_name == "TAG":
                conv = TAGConv(in_dim if i == 0 else graph_hidden_channels, graph_hidden_channels)
            elif model_name == "Cheb":
                conv = ChebConv(in_dim if i == 0 else graph_hidden_channels, graph_hidden_channels, K=2)
            elif model_name == "SGC":
                conv = SGConv(in_dim if i == 0 else graph_hidden_channels, graph_hidden_channels, K=2)
            elif model_name == "Transformer":
                conv = TransformerConv(in_dim if i == 0 else graph_hidden_channels, graph_hidden_channels // heads,
                                       heads=heads)
            else:
                conv = GraphConv(in_dim if i == 0 else graph_hidden_channels, graph_hidden_channels)

            convs.append(conv)
            if graph_norm == "BatchNorm":
                norms.append(nn.BatchNorm1d(graph_hidden_channels))
            elif graph_norm == "GraphNorm":
                norms.append(GraphNorm(graph_hidden_channels))
            else:
                norms.append(LayerNorm(graph_hidden_channels, mode="node"))

        self._convs = nn.ModuleList(convs)
        self._norms = nn.ModuleList(norms)
        self._act = getattr(F, activation, F.relu)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv, norm in zip(self._convs, self._norms):
            h = conv(x, edge_index)
            h = norm(h)
            h = self._act(h)
            if self.residual and h.shape == x.shape:
                x = x + h
            else:
                x = h
        return x


class GNNTester:
    def __init__(self, train_data=None):
        self.train_data = train_data

    def create_smart_parameter_combinations(self, max_combinations=50):
        models = ["GCN", "SAGE", "GIN", "GAT", "TAG", "SGC", "Transformer", "Graph", "Cheb"]
        layers = [2, 3, 4, 5, 6]  # додадено 4, 6
        hidden = [32, 64, 128, 256, 512]  # додадено 256, 512
        norms = ["BatchNorm", "GraphNorm", "LayerNorm"]
        activations = ["relu", "gelu", "leaky_relu", "elu", "selu", "swish"]  # додадено
        heads = [1, 2, 4, 8]  # за GAT/Transformer
        aggrs = ["mean", "max", "add"]  # различни aggregation
        learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
        weight_decays = [1e-5, 1e-4, 1e-3]
        dropouts = [0.1, 0.2, 0.3, 0.5]

        combos = []
        for m, l, h, n, a, lr, wd, drop in itertools.product(
                models, layers, hidden, norms, activations,
                learning_rates, weight_decays, dropouts
        ):
            # За GAT/Transformer изберете случајни heads
            head_count = random.choice(heads) if m in ["GAT", "Transformer"] else 1
            aggr_method = random.choice(aggrs)

            combos.append({
                "model_name": m,
                "graph_layers": l,
                "graph_hidden_channels": h,
                "graph_norm": n,
                "activation": a,
                "heads": head_count,
                "aggr": aggr_method,
                "residual": random.choice([True, False]),  # randomize residual
                "learning_rate": lr,
                "weight_decay": wd,
                "dropout": drop
            })

        random.shuffle(combos)
        return combos[:max_combinations]

