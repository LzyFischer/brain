import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class AE(nn.Module):
    def __init__(self, net_params, args):
        """
        An MLP-based anomaly detection model with an encoder-decoder structure.
        
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units in each layer.
            bottleneck_dim (int): Dimensionality of the bottleneck layer.
        """
        super().__init__()
        input_dim = net_params["in_channels"]
        hidden_dim = net_params["hidden_channels"]
        bottleneck_dim = net_params["hidden_channels"]

        # Encoder: Compress the input into a bottleneck representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU()
        )
        
        # Decoder: Reconstruct the input from the bottleneck representation
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x , data.edge_index, data.batch
        bottleneck = self.encoder(x)
        reconstruction = self.decoder(bottleneck)

        # global mean
        reconstruction = global_mean_pool(reconstruction, batch)
        
        return reconstruction