import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINConv,
    global_mean_pool,
    global_max_pool,
    GCNConv,
    SAGEConv,
    GATConv,
    GatedGraphConv,
    SGConv,
    ResGatedGraphConv,
)

import pdb
from torch_geometric.nn import TransformerConv


class GIN_pyg(nn.Module):
    def __init__(self, net_params, args):
        super(GIN_pyg, self).__init__()
        in_channels = net_params["in_channels"]
        hidden_channels = net_params["hidden_channels"]
        out_channels = net_params["out_channels"]
        num_layers = net_params["num_layers"]
        dropout = net_params["dropout"]

        self.readout_type = net_params["readout"]
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        # self.conv1 = GINConv(
        #     nn.Sequential(
        #         nn.Linear(in_channels, hidden_channels),
        #         # nn.ReLU(),
        #         # nn.Linear(hidden_channels, hidden_channels),
        #     )
        # )
        # self.conv2 = GINConv(
        #     nn.Sequential(
        #         nn.Linear(hidden_channels, hidden_channels),
        #         # nn.ReLU(),
        #         # nn.Linear(hidden_channels, hidden_channels),
        #     )
        # )

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.fc_encode = nn.Linear(self.in_channels, self.hidden_channels)
        self.fc = nn.Linear(self.hidden_channels, 1)

        self.conv1_SC = GCNConv(in_channels, hidden_channels)
        self.conv2_SC = GCNConv(hidden_channels, hidden_channels)

        self.fc_encode_SC = nn.Linear(self.in_channels, self.hidden_channels)
        self.fc_SC = nn.Linear(self.hidden_channels, 1)

    def forward(self, data):
        x, x_SC, edge_index, edge_weight, batch = (
            data.x,
            data.x_SC,
            data.edge_index_SC,
            data.edge_weight,
            data.batch,
        )
        # pdb.set_trace()
        x = F.relu(self.conv1(x, edge_index)) + F.relu(self.fc_encode(x))
        # pdb.set_trace()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index)) + x
        x = F.dropout(x, p=self.dropout, training=self.training)

        x_SC = F.relu(self.conv1_SC(x_SC, edge_index)) + F.relu(self.fc_encode_SC(x_SC))
        x_SC = F.dropout(x_SC, p=self.dropout, training=self.training)
        x_SC = F.relu(self.conv2_SC(x_SC, edge_index)) + x_SC
        x_SC = F.dropout(x_SC, p=self.dropout, training=self.training)

        x = x + x_SC

        x = global_mean_pool(x, batch)
        predict = self.fc(x)
        # pdb.set_trace()

        return predict
