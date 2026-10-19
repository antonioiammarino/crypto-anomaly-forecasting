import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv


class GATBaseline(nn.Module):
    # Keep the layer sizes configurable because they are selected inside the
    # nested training loop rather than fixed before model comparison.
    def __init__(
        self,
        input_channels=5,
        hidden_channels=16,
        attention_heads=4,
        classes=3,
        dropout=0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Multiple heads let the first layer learn different neighbour weights.
        # Their outputs are concatenated before the second attention layer.
        self.first_attention = GATConv(
            input_channels,
            hidden_channels,
            heads=attention_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )
        expanded_channels = hidden_channels * attention_heads
        self.normalization = nn.LayerNorm(expanded_channels)
        self.second_attention = GATConv(
            expanded_channels,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )
        self.classifier = nn.Linear(hidden_channels, classes)

    # Apply message passing only through the edges selected for the experiment.
    def forward(self, node_features, edge_index):
        hidden = F.dropout(node_features, p=self.dropout, training=self.training)
        hidden = self.first_attention(hidden, edge_index)
        hidden = self.normalization(hidden)
        hidden = F.elu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = self.second_attention(hidden, edge_index)
        hidden = F.elu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.classifier(hidden)


# Use the same temporal input information without graph message passing. This gives
# the GAT comparison a neural baseline with comparable depth and training settings.
class MLPBaseline(nn.Module):
    # Mirror the GAT depth and hidden size so the comparison does not give one
    # architecture a larger neural network by construction.
    def __init__(
        self,
        input_channels=5,
        hidden_channels=16,
        classes=3,
        dropout=0.3,
    ):
        super().__init__()
        self.dropout = dropout
        self.first_layer = nn.Linear(input_channels, hidden_channels)
        self.normalization = nn.LayerNorm(hidden_channels)
        self.second_layer = nn.Linear(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, classes)

    # Keep edge_index in the function so GAT and MLP can share the training loop.
    # The MLP intentionally ignores it and uses only the causal feature window.
    def forward(self, node_features, edge_index=None):
        del edge_index
        hidden = F.dropout(node_features, p=self.dropout, training=self.training)
        hidden = self.first_layer(hidden)
        hidden = self.normalization(hidden)
        hidden = F.elu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = self.second_layer(hidden)
        hidden = F.elu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.classifier(hidden)


# Share two GAT layers across all assets while keeping one classifier per asset.
# Temporal and relational edges deliberately use the same attention operator, so
# the experiment isolates the effect of adding the relational topology.
class PanelGAT(nn.Module):
    def __init__(
        self,
        input_channels=4,
        hidden_channels=16,
        attention_heads=4,
        number_of_assets=17,
        classes=3,
        dropout=0.3,
    ):
        super().__init__()
        self.dropout = dropout
        self.classes = classes
        self.first_attention = GATConv(
            input_channels,
            hidden_channels,
            heads=attention_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )
        expanded_channels = hidden_channels * attention_heads
        self.normalization = nn.LayerNorm(expanded_channels)
        self.second_attention = GATConv(
            expanded_channels,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )

        # ModuleList is required so PyTorch registers every asset-specific head.
        self.classifiers = nn.ModuleList()
        for _ in range(number_of_assets):
            self.classifiers.append(nn.Linear(hidden_channels, classes))

    # Route every node through the classifier belonging to its cryptocurrency.
    def forward(self, node_features, edge_index, node_asset_indices):
        hidden = F.dropout(node_features, p=self.dropout, training=self.training)
        hidden = self.first_attention(hidden, edge_index)
        hidden = self.normalization(hidden)
        hidden = F.elu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = self.second_attention(hidden, edge_index)
        hidden = F.elu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)

        logits = hidden.new_empty((len(hidden), self.classes))
        for asset, classifier in enumerate(self.classifiers):
            mask = node_asset_indices == asset
            if torch.any(mask):
                logits[mask] = classifier(hidden[mask])
        return logits
