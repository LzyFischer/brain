import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, GCNConv, SAGEConv, GATConv, GatedGraphConv, SGConv, ResGatedGraphConv 

import pdb
from torch_geometric.nn import TransformerConv


class GIN_pyg(nn.Module):
    def __init__(self, net_params, args):
        super(GIN_pyg, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = net_params["in_channels"]
        hidden_channels = net_params["hidden_channels"]
        out_channels = net_params["out_channels"]
        num_layers = net_params["num_layers"]
        self.readout_type = net_params["readout"]
        dropout = net_params["dropout"]
        self.modality = args.modality

        model = GCNConv

        if net_params["activation"] == "elu":
            activation = nn.ELU()
        else:
            activation = nn.ReLU()

        nnlayer = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            activation,
        )
        """modify"""
        # self.layers.append(model(nnlayer))
        # self.feature_linear = nn.Linear(33, hidden_channels)
        self.x_linear = nn.Linear(in_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.layers.append(model(in_channels, hidden_channels))
        for i in range(num_layers):
            self.layers.append(model(hidden_channels, hidden_channels))
            # self.layers.append(activation)
        self.predict = nn.Linear(hidden_channels, net_params["n_vars"])
        
        # self.dropout = nn.Dropout(dropout)

        # for i in range(1, num_layers):
        #     if i != num_layers - 1:
        #         nnlayer = nn.Sequential(
        #             nn.Linear(hidden_channels, hidden_channels), activation
        #         )
        #         self.layers.append(model(nnlayer))
        #     else:
        #         nnlayer = nn.Sequential(
        #             nn.Linear(hidden_channels, out_channels), activation
        #         )
        #         self.layers.append(model(nnlayer))
        # self.predict = nn.Linear(
        #     in_features=out_channels, out_features=net_params["n_vars"]
        # )
        """modify end"""

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, GINConv):
                for layer in m.nn:
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
        
        # self.input_transform = nn.Sequential(
        #     nn.Linear(in_channels, hidden_channels),
        #     activation,
        # )

        # # Adding TransformerConv layers
        # for i in range(num_layers):
        #     if i != num_layers - 1:
        #         self.layers.append(TransformerConv(hidden_channels, hidden_channels, heads=1, concat=True))
        #     else:
        #         self.layers.append(TransformerConv(hidden_channels, out_channels, heads=1, concat=True))
       

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # x = torch.ones_like(x).to(x.device)

        """modify"""
        # feature = data.feature
        # x = self.x_linear(x)
        # feature = self.feature_linear(feature)
        # feature = feature.unsqueeze(1).repeat(1, x.size(0) //feature.size(0), 1).reshape(-1, feature.size(1))
        # feature = torch.zeros_like(feature).to(feature.device)
        # x = x + feature
        # x = F.relu(x)

        # x = self.input_transform(x)
        # for i, layer in enumerate(self.layers):
        #     if isinstance(layer, TransformerConv):
        #         x = layer(x, edge_index) + x
        #         # x_sc = layer(x_sc, edge_index_sc)
        #     else:
        #         x = layer(x, edge_index)
        #         # x_sc = layer(x_sc, edge_index_sc)
            
        #     if i != len(self.layers) - 1:  # Don't apply dropout after the last layer
        #         x = self.dropout(x)
        #         # x_sc = self.dropout(x_sc)

        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x, edge_index)
            else:
                x = layer(x, edge_index) + x
            x = F.relu(x)
            if i != len(self.layers) - 1:  # Don't apply dropout after the last layer
                x = self.dropout(x)


        if self.readout_type == "mean":
            x = global_mean_pool(x, batch)
        elif self.readout_type == "max":
            x = global_max_pool(x, batch)
        else:
            raise ValueError(f"Invalid readout type: {self.readout_type}")
        

        # x_sc, edge_index_sc, batch_sc = data.x_SC, data.edge_index_SC, data.batch
        # for i, layer in enumerate(self.layers):
        #     x_sc = layer(x_sc, edge_index_sc)
        #     if i != len(self.layers) - 1:
        #         x_sc = self.dropout(x_sc)
        # if self.readout_type == "mean":
        #     x_sc = global_mean_pool(x_sc, batch_sc)
        # elif self.readout_type == "max":
        #     x_sc = global_max_pool(x_sc, batch_sc)

        if self.modality == "SC":
            x = x_sc
        elif self.modality == "FC":
            x = x
        elif self.modality == "Both":
            x += x_sc

        """modify end"""

        predict = self.predict(x)
        # pdb.set_trace()
        return predict
