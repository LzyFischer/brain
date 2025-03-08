import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool


class MLP_pyg(nn.Module):
    def __init__(self, net_params, args):
        super(MLP_pyg, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = net_params["in_channels"]
        hidden_channels = net_params["hidden_channels"]
        num_layers = net_params["num_layers"]
        self.readout_type = net_params["readout"]
        dropout = net_params["dropout"]
        self.modality = args.modality

        # Choose activation function
        if net_params["activation"] == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()

        # Input layer
        self.x_linear = nn.Linear(in_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)

        # Hidden layers
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))

        # Output layer
        self.predict = nn.Linear(hidden_channels, net_params["n_vars"])

    def forward(self, data):
        x, batch = data.x, data.batch

        # Input layer
        x = self.x_linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation(x)
            if i != len(self.layers) - 1:  # Skip dropout after the last layer
                x = self.dropout(x)

        # Readout
        if self.readout_type == "mean":
            x = global_mean_pool(x, batch)
        elif self.readout_type == "max":
            x = global_max_pool(x, batch)
        else:
            raise ValueError(f"Invalid readout type: {self.readout_type}")

        # Modality handling
        if self.modality == "SC":
            x = x_sc  # Assume `x_sc` is externally provided
        elif self.modality == "FC":
            pass
        elif self.modality == "Both":
            x += x_sc

        # Final prediction
        predict = self.predict(x)
        return predict