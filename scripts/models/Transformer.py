import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, GCNConv

import pdb


class Transformers_(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = net_params["in_channels"]
        hidden_channels = net_params["hidden_channels"]
        out_channels = net_params["out_channels"]
        num_layers = net_params["num_layers"]
        self.readout_type = net_params["readout"]
        dropout = net_params["dropout"]

        if net_params["activation"] == "elu":
            activation = nn.ELU()
        else:
            activation = nn.ReLU()

        nnlayer = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            activation,
        )
        self.layers.append(GINConv(nnlayer))
        self.dropout = nn.Dropout(dropout)

        for i in range(1, num_layers):
            if i != num_layers - 1:
                nnlayer = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels), activation
                )
                self.layers.append(GINConv(nnlayer))
            else:
                nnlayer = nn.Sequential(
                    nn.Linear(hidden_channels, out_channels), activation
                )
                self.layers.append(GINConv(nnlayer))
        self.predict = nn.Linear(
            in_features=out_channels, out_features=net_params["n_vars"]
        )


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index_FC, data.batch
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != len(self.layers) - 1:  # Don't apply dropout after the last layer
                x = self.dropout(x)
                x = F.relu(x)

        if self.readout_type == "mean":
            x = global_mean_pool(x, batch)
        elif self.readout_type == "max":
            x = global_max_pool(x, batch)
        else:
            raise ValueError(f"Invalid readout type: {self.readout_type}")
        
        x_sc, edge_index_sc, batch_sc = data.x_SC, data.edge_index_SC, data.batch
        for i, layer in enumerate(self.layers):
            x_sc = layer(x_sc, edge_index_sc)
            if i != len(self.layers) - 1:
                x_sc = self.dropout(x_sc)
                x_sc = F.relu(x_sc)
        if self.readout_type == "mean":
            x_sc = global_mean_pool(x_sc, batch_sc)
        elif self.readout_type == "max":
            x_sc = global_max_pool(x_sc, batch_sc)

        x += x_sc

        predict = torch.sigmoid(self.predict(x))
        return predict
