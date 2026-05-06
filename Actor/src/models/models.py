"""
All five models for Milestone 2.
Every model has the same signature: forward(x, edge_index=None)
so the training loop doesn't need to know which model it's running.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, APPNP, GATConv


class LogisticRegression(nn.Module):
    """Linear classifier — no hidden layers, no graph. Simplest possible baseline."""

    def __init__(self, in_features, num_classes, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x, edge_index=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.linear(x)


class MLP(nn.Module):
    """Nonlinear feature-only model. No graph."""

    def __init__(self, in_features, num_classes, hidden_dim, num_layers, dropout):
        super().__init__()
        assert num_layers >= 2, "MLP needs at least 2 layers (one hidden, one output)"
        self.dropout = dropout

        dims = [in_features] + [hidden_dim] * (num_layers - 1) + [num_classes]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x, edge_index=None):
        for i, layer in enumerate(self.layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x)
            if i < len(self.layers) - 1:   # no relu/dropout after last layer
                x = F.relu(x)
        return x


class GCN(nn.Module):
    """Standard GCN. GCNConv adds self-loops and symmetric normalisation by default."""

    def __init__(self, in_features, num_classes, hidden_dim, num_layers, dropout):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_features, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, num_classes))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class GraphSAGE(nn.Module):
    """
    GraphSAGE. SAGEConv concatenates the node's own embedding with the
    aggregated neighbour embedding, giving a stronger self-identity signal.
    This can help on heterophilic graphs where neighbour labels differ.
    """

    def __init__(self, in_features, num_classes, hidden_dim, num_layers, dropout, aggr="mean"):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_features, hidden_dim, aggr=aggr))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
        self.convs.append(SAGEConv(hidden_dim, num_classes, aggr=aggr))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class GAT(nn.Module):
    """
    GAT — Graph Attention Network. Each edge gets a learned attention weight,
    so neighbours are not treated equally. On a heterophilic graph this gives
    the model a chance to down-weight unhelpful neighbours, though the attention
    still happens within a 1-hop neighbourhood.
    """

    def __init__(self, in_features, num_classes, hidden_dim, num_layers,
                 heads, dropout):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout

        # Intermediate layers use multi-head concat (hidden_dim * heads features).
        # The last layer averages heads (concat=False) so the output has num_classes dims.
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_features, hidden_dim, heads=heads,
                                   dropout=dropout, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads,
                                       dropout=dropout, concat=True))
        self.convs.append(GATConv(hidden_dim * heads, num_classes, heads=1,
                                   dropout=dropout, concat=False))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)  # GAT paper uses ELU
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class APPNPNet(nn.Module):
    """
    APPNP: first transforms features with an MLP, then propagates using
    personalised PageRank. The alpha (teleport) parameter controls how much
    of the original node representation is retained at each step.
    High alpha = less smoothing = better for heterophilic graphs.
    """

    def __init__(self, in_features, num_classes, hidden_dim, mlp_layers, dropout, K, alpha):
        super().__init__()
        self.dropout = dropout

        # Build the MLP: in -> hidden x mlp_layers -> out
        dims = [in_features] + [hidden_dim] * mlp_layers + [num_classes]
        self.lins = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.lins.append(nn.Linear(dims[i], dims[i + 1]))

        # Personalised PageRank propagation
        self.prop = APPNP(K=K, alpha=alpha)

    def forward(self, x, edge_index):
        # MLP pass
        for i, lin in enumerate(self.lins):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.relu(x)
        # Graph propagation
        x = self.prop(x, edge_index)
        return x
