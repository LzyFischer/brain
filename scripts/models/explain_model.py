import torch
import pdb
from tqdm import tqdm
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.data import Data



class GCNConvPerturb(GCNConv):
    """GCNConv layer with edge perturbation capability"""
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight=None, P_hat=None):
        if P_hat is not None:
            edge_weight = P_hat.to(x.device)
        return super().forward(x, edge_index, edge_weight)

class GIN_pyg_perturb(nn.Module):
    """GIN model with edge perturbation capability"""
    def __init__(self, base_model):

        super().__init__()
        self.gc1 = GCNConvPerturb(base_model.conv1.in_channels, base_model.conv1.out_channels)
        self.gc2 = GCNConvPerturb(base_model.conv2.in_channels, base_model.conv2.out_channels)
        self.fc = nn.Linear(base_model.fc.in_features, base_model.fc.out_features)
        
        # Initialize weights from base model
        self.gc1.load_state_dict(base_model.conv1.state_dict())
        self.gc2.load_state_dict(base_model.conv2.state_dict())
        self.fc.load_state_dict(base_model.fc.state_dict())
        
        # P_hat parameters
        self.P_vec = None
        self.edge_additions = True  # Set to False for edge deletions

    def reset_parameters(self, edge_index_size):
        num_edges = edge_index_size[1]
        self.P_vec = Parameter(torch.FloatTensor(num_edges))
        
        # Initialize with small positive values for better gradient flow
        if self.edge_additions:
            nn.init.uniform_(self.P_vec, 0.0, 0.1)  # Start with low probability additions
        else:
            nn.init.constant_(self.P_vec, 5.0)  # High initial value for deletions (sigmoid(5) ≈ 1)

    def forward(self, data, P_hat=None):
        x, edge_index, edge_weight, batch = (
            data.x,
            data.edge_index,
            data.edge_weight,
            data.batch,
        )

        x1 = F.relu(self.gc1(x, edge_index, edge_weight, P_hat))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        
        x2 = F.relu(self.gc2(x1, edge_index, edge_weight, P_hat))
        x2 = F.dropout(x2, p=0.5, training=self.training)
        
        x = global_mean_pool(x2, batch)
        return self.fc(x)


class BrainCFExplainer:
    def __init__(self, model, lr=0.1, num_epochs=200, lambda_reg=0.00001):
        self.model = model
        self.lr = lr
        self.num_epochs = num_epochs
        self.lambda_reg = lambda_reg

    def explain(self, data, target_class=None):
        device = next(self.model.parameters()).device
        data = data.to(device)
        self.model.eval()
        
        # Get original prediction
        with torch.no_grad():
            original_out = self.model(data)
            original_pred = torch.sigmoid(original_out).item()
        
        # Determine target class if not specified
        if target_class is None:
            target_class = 1 - round(original_pred)
        target = torch.tensor([target_class], dtype=torch.float32, device=device)
        
        # Initialize perturbation parameter
        delta = torch.nn.Parameter(torch.zeros_like(data.x, device=device))
        optimizer = torch.optim.Adam([delta], lr=self.lr)
        
        # Optimize perturbation
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            
            # Apply perturbation
            perturbed_x = data.x + delta
            perturbed_data = Data(
                x=perturbed_x,
                edge_index=data.edge_index,
                batch=data.batch,
                edge_weight=data.edge_weight if hasattr(data, 'edge_weight') else None
            )
            
            # Get prediction
            out = self.model(perturbed_data)
            
            # Calculate losses
            loss = F.binary_cross_entropy_with_logits(out.squeeze(0), target)
            reg_loss = self.lambda_reg * torch.norm(delta, p=1)
            total_loss = loss + reg_loss
            
            # Optimize
            total_loss.backward()
            optimizer.step()
        
        # Generate counterfactual data
        cf_x = data.x + delta.detach()
        cf_data = Data(
            x=cf_x,
            edge_index=data.edge_index,
            batch=data.batch,
            edge_weight=data.edge_weight if hasattr(data, 'edge_weight') else None
        )
        
        # Get final prediction
        with torch.no_grad():
            cf_pred = torch.sigmoid(self.model(cf_data)).item()
        
        return {
            "original_data": data,
            "cf_data": cf_data,
            "original_pred": original_pred,
            "cf_pred": cf_pred,
            "perturbation": delta.detach()
        }


class SurrogateGCN(nn.Module):
    def __init__(self, net_params, args):
        super(SurrogateGCN, self).__init__()
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

        # Original GCN components
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc_encode = nn.Linear(self.in_channels, self.hidden_channels)
        self.fc = nn.Linear(self.hidden_channels, 1)

    def forward(self, x, edge_index):
        # Replicate original edge_weight handling
        edge_weight = torch.zeros(edge_index.size(1)).to(x.device)
        batch = edge_index.new_zeros(x.size(0))
        
        # First GCN layer
        x_conv = F.relu(self.conv1(x, edge_index))
        x_fc = F.relu(self.fc_encode(x))
        x = x_conv + x_fc
        
        # Dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GCN layer with residual connection
        x_conv2 = F.relu(self.conv2(x, edge_index))
        x = x_conv2 + x
        
        # Final dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Global mean pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            # Handle single graph case
            x = x.mean(dim=0, keepdim=True)
            
        # Final prediction
        predict = self.fc(x)
        predict = torch.sigmoid(predict)
        return predict





def explain(model, test_set, args):
    indices = np.load('dataset/processed/att_region.pkl', allow_pickle=True)
    max_index = 378
    region_mask = np.zeros(max_index + 1, dtype=bool)
    region_mask[indices] = True
    def visualize_perturbation(delta, name):
        # delta = explanation["perturbation"].cpu().numpy()
        print("Top 20 perturbed features:")
        # delta = delta
        for idx in np.argsort(-np.abs(delta).max(0))[:20]:
            # print(f"Feature {idx}: Max perturbation {delta[:, idx].max():.4f}")
            print(f"{idx}, {delta[:, idx].max():.4f}")
            print(f"{idx}, {delta[:, idx].max():.4f}", file=open(f'logs/delta_{args.seed}_{name}_f.txt', "a"))
        
        # plot line
        import matplotlib.pyplot as plt
        delta_value = np.abs(delta).max(0)
        plt.plot(delta_value)
        plt.savefig(f"plot/importance_{args.seed}_{name}_f.png")
        plt.close()
        plt.plot(delta_value*region_mask)
        plt.savefig(f"plot/importance_mask_{args.seed}_{name}_f.png")
        plt.close()
        plt.imshow(delta)
        plt.savefig(f"plot/heatmap_{args.seed}_{name}_f.png")
        plt.close()
        print(f"{delta_value.mean()}, {(delta_value*region_mask).sum() / len(indices)}", file=open(f'logs/delta_{args.seed}_{name}_f.txt', "a"))

    # ########## BrainCFExplainer #############
    explainer = BrainCFExplainer(model, lr=0.05, lambda_reg=0.000001)

    # Get a sample from your test set
    delta_all = None
    for sample_id in tqdm(range(len(test_set))):
        sample_data = test_set[sample_id].to(device)

        # Generate explanation
        explanation = explainer.explain(sample_data)
        delta = np.abs(explanation["perturbation"].cpu().numpy())
        delta_all = delta if delta_all is None else delta_all + delta

        print(f"Original prediction: {explanation['original_pred']:.4f}")
        print(f"Counterfactual prediction: {explanation['cf_pred']:.4f}")
            
    visualize_perturbation(delta_all, "cfexplainer")

    ########### CaptumExplainer
    # Get prediction
    mask_all = None
    for sample_id in tqdm(range(len(test_set))):
        sample_data = test_set[sample_id].to(device)
        surrogate_model = SurrogateGCN(config["net_params"], args)
        surrogate_model.load_state_dict(model.state_dict())
        surrogate_model = surrogate_model.to(device)         # Batch indices

        captum_explainer = Explainer(
            model=surrogate_model,
            algorithm=CaptumExplainer('IntegratedGradients'),
            explanation_type='model',
            model_config=dict(
                mode='binary_classification',
                task_level='graph',
                return_type='probs',
            ),
            node_mask_type='attributes',
            edge_mask_type=None,
        )
        mask = captum_explainer(sample_data.x.to(device), sample_data.edge_index.to(device)).node_mask
        mask = mask.cpu().numpy()
        mask_all = mask if mask_all is None else mask_all + mask
    visualize_perturbation(mask_all, 'IntegratedGradients')


    ########### CaptumExplainer
    # Get prediction
    mask_all = None
    for sample_id in tqdm(range(len(test_set))):
        sample_data = test_set[sample_id].to(device)
        surrogate_model = SurrogateGCN(config["net_params"], args)
        surrogate_model.load_state_dict(model.state_dict())
        surrogate_model = surrogate_model.to(device)         # Batch indices

        captum_explainer = Explainer(
            model=surrogate_model,
            algorithm=CaptumExplainer('Saliency'),
            explanation_type='model',
            model_config=dict(
                mode='binary_classification',
                task_level='graph',
                return_type='probs',
            ),
            node_mask_type='attributes',
            edge_mask_type=None,
        )
        mask = captum_explainer(sample_data.x.to(device), sample_data.edge_index.to(device)).node_mask
        mask = mask.cpu().numpy()
        mask_all = mask if mask_all is None else mask_all + mask
    visualize_perturbation(mask_all, 'Saliency')

    ########### CaptumExplainer
    # Get prediction
    mask_all = None
    for sample_id in tqdm(range(len(test_set))):
        sample_data = test_set[sample_id].to(device)
        surrogate_model = SurrogateGCN(config["net_params"], args)
        surrogate_model.load_state_dict(model.state_dict())
        surrogate_model = surrogate_model.to(device)         # Batch indices

        captum_explainer = Explainer(
            model=surrogate_model,
            algorithm=CaptumExplainer('GuidedBackprop'),
            explanation_type='model',
            model_config=dict(
                mode='binary_classification',
                task_level='graph',
                return_type='probs',
            ),
            node_mask_type='attributes',
            edge_mask_type=None,
        )
        mask = captum_explainer(sample_data.x.to(device), sample_data.edge_index.to(device)).node_mask
        mask = mask.cpu().numpy()
        mask_all = mask if mask_all is None else mask_all + mask
    visualize_perturbation(mask_all, 'GuidedBackprop')
    

    ########### GNNExplainer
    mask_all = None
    for sample_id in tqdm(range(len(test_set))):
        sample_data = test_set[sample_id].to(device)
        surrogate_model = SurrogateGCN(config["net_params"], args)
        surrogate_model.load_state_dict(model.state_dict())
        surrogate_model = surrogate_model.to(device) 

        gnn_explainer = Explainer(
            model=surrogate_model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            model_config=dict(
                mode='binary_classification',
                task_level='graph',
                return_type='probs',
            ),
            node_mask_type='attributes',
            edge_mask_type=None,
        )

        mask = captum_explainer(sample_data.x.to(device), sample_data.edge_index.to(device)).node_mask
        mask = mask.cpu().numpy()
        mask_all = mask if mask_all is None else mask_all + mask
    visualize_perturbation(mask_all, 'gnnexplainer')