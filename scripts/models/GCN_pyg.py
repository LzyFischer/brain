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
    GraphConv
)
import numpy as np
import matplotlib.pyplot as plt

import pdb
from torch_geometric.nn import TransformerConv
from utils.utils import plot_edge_weight

from torch_geometric.nn import MessagePassing


class GCN_pyg(nn.Module):
    def __init__(self, net_params, args):
        super(GCN_pyg, self).__init__()
        self.args = args
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
        #     )
        # )
        # self.conv2 = GINConv(
        #     nn.Sequential(
        #         nn.Linear(hidden_channels, hidden_channels),
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

        self.register_buffer("node_mask", torch.from_numpy(np.load('./dataset/processed/att_region_mask.pkl', allow_pickle=True)).float())
        self.learnable_weights = nn.Parameter(torch.ones_like(self.node_mask))
        self.fixed_weights = self.node_mask.clone()

        self.register_buffer("edge_mask", torch.from_numpy(np.load('./dataset/processed/att_region_adj.pkl', allow_pickle=True)).float())
        self.learnable_edge_weights = nn.Parameter(torch.zeros_like(self.edge_mask))
        self.fixed_edge_weights = self.edge_mask.clone()


    def forward(self, data):
        x, edge_index, edge_weight, batch = (
            data.x,          # [num_nodes, feature_size]
            data.edge_index, # [2, num_edges]
            data.edge_weight, # [num_edges]
            data.batch,      # [num_nodes] (mapping nodes to their respective graphs)
        )


        # self.plot_edge(edge_index, edge_weight)
        # Applying mask to edge and node features
        edge_weight = self.get_learnable_edge_weights(edge_index, edge_weight)
        # x = self.get_learnable_weights(batch, x)
        
        x = F.relu(self.conv1(x, edge_index, edge_weight)) + F.relu(self.fc_encode(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Applying learnable and fixed weights to nodes during message passing
        x = F.relu(self.conv2(x, edge_index, edge_weight)) + x
        x = F.dropout(x, p=self.dropout, training=self.training)


        # Global mean pooling across batches
        x = global_mean_pool(x, batch)
        predict = self.fc(x)

        return predict
    
    def plot_edge(self, edge_index, edge_weight):
        edge_index_first = edge_index[:, edge_index[0] < 379]
        edge_index_first = edge_index_first[:, edge_index_first[1] < 379]

        adj = torch.zeros((379, 379))
        adj[edge_index_first[0], edge_index_first[1]] = edge_weight.cpu()[:edge_index_first.shape[1]]

        # use plot_edge_weight function to plot the adj
        plot_edge_weight(adj, adj>0, 0, 0)
        pdb.set_trace()


    def get_learnable_edge_weights(self, edge_index, edge_weight):
        local_i = edge_index[0] % 379
        local_j = edge_index[1] % 379
        fixed_edge_mask = self.edge_mask[local_i, local_j]
        learnable_edge_weights = (self.learnable_edge_weights + self.learnable_edge_weights.T) / 2
        learnable_edge = learnable_edge_weights[local_i, local_j]
        learnable_edge = torch.sigmoid(learnable_edge) * 2
        effective_edge_mask = fixed_edge_mask * learnable_edge + (1 - fixed_edge_mask)
        effective_edge_mask = learnable_edge
        if self.args.original_edge_weight:
            edge_weight = effective_edge_mask * edge_weight
        else:
            edge_weight = effective_edge_mask
        return edge_weight
    
    def get_learnable_weights(self, batch, x):
        # Extract the mask values for each node
        learnable_part = self.node_mask * self.learnable_weights
        fixed_part = 1 - self.node_mask

        learnable_part = learnable_part.repeat(batch.max().item() + 1).unsqueeze(1)
        fixed_part = fixed_part.repeat(batch.max().item() + 1).unsqueeze(1)
        # Apply the mask to the node features
        x = x * (learnable_part + fixed_part)
        return x