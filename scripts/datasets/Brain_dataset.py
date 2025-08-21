# -*- coding: utf-8 -*-
# @Time    : 9/27/2023 7:42 PM
# @Author  : Gang Qu
# @FileName: HCDP_dataset.py

import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data

# global mean pooling and global max pooling
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_sparse import SparseTensor
import os
import pickle
import sys
import dgl
import numpy as np
import pdb
from tqdm import tqdm
from torch_geometric.data import DataLoader
import torch.nn as nn
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, accuracy_score


# from utils.utils import get_spectral_embedding
from dgl.data import DGLDataset

from typing import Any, Dict, List, Tuple

# set the path to the root directory of the project
sys.path.append(os.path.abspath("../"))




class Brain(InMemoryDataset):
    def __init__(
        self,
        task,
        x_attributes=None,
        processed_path="./data/processed",
        rawdata_path=None,
        suffix=None,
        args=None,
    ):

        if suffix is None:
            suffix = f"{args.data_name}" if args and hasattr(args, "data_name") else ""
        self.processed_path = os.path.join(processed_path, f"{suffix}.pt")

        self.task = task
        self.x_attributes = x_attributes
        self.rawdata_path = rawdata_path
        self.suffix = suffix

        super().__init__(pre_transform=self.make_pre_transform(args))

        self.data, self.slices = torch.load(self.processed_path)

        """modify"""
        if args is not None and hasattr(args, "label_index") and args.label_index is not None and len(self.data['y'].shape) > 1 and self.data["y"].shape[1] > 1:
            self.data.y = self.data["y"][:, [args.label_index]] 
        else:
            self.data.y = self.data["y"].unsqueeze(1) 
        """modify end"""
    
    def make_pre_transform(self, args):
        def transform(data: Dict[str, Any]) -> Data:
            x = torch.tensor(data["FC"], dtype=torch.float32)
            x_SC = torch.tensor(data["SC"], dtype=torch.float32)
            x_SC = ((x_SC - x_SC.min()) / (x_SC.max() - x_SC.min()) * 1e5).log1p()

            threshold = args.threshold if args and hasattr(args, "threshold") else 0
            edge_index_FC = (x > threshold).nonzero().t().contiguous()
            edge_index_FC = edge_index_FC[:, edge_index_FC[0] != edge_index_FC[1]]
            row, col = edge_index_FC
            edge_weight_FC = x[row, col]

            edge_index_SC = (x_SC >= 7).nonzero().t().contiguous()
            edge_index_SC = edge_index_SC[:, edge_index_SC[0] != edge_index_SC[1]]
            row, col = edge_index_SC
            edge_weight_SC = torch.tensor(x_SC[row, col], dtype=torch.float32)

            feature = (
                torch.tensor(data["feature"], dtype=torch.float32).unsqueeze(0)
                if "feature" in data.keys()
                else None
            )
            label_tensor = torch.tensor(data["label"], dtype=torch.float32).unsqueeze(0)

            return Data(
                x=x,
                x_SC=x_SC,
                edge_index=edge_index_FC,
                edge_weight=edge_weight_FC,
                edge_index_SC=edge_index_SC,
                edge_weight_SC=edge_weight_SC,
                y=label_tensor,
                feature=feature,
            )
        return transform

    def processed_file_names(self):
        return os.path.basename(self.processed_path)

    def process(self) -> None:
        with open(self.rawdata_path, "rb") as f:
            data = pickle.load(f)

        data_list = []
        for i in tqdm(range(len(data))):
            if self.pre_transform is not None:
                data_list.append(self.pre_transform(data[i]))

        self.data, self.slices = self.collate(data_list)
        print("Saving...")
        torch.save((self.data, self.slices), self.processed_path)

    @property
    def processed_dir(self):
        return os.path.dirname(self.processed_path)

    def process_data(self, data):
        data_list = []
        for i in range(len(data)):
            try:
                data_list.append(self.pre_transform(data[i]))
            except:
                pdb.set_trace()

        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices), self.processed_path)


if __name__ == "__main__":

    dataset = Brain(
        task="classification",
        x_attributes=["adj"],
        processed_path="/project/uvadm/zhenyu/project/brain/data/processed",
        rawdata_path="/project/uvadm/zhenyu/project/brain/dataset/all_graphs_2.pkl",
    )
    train_test_split = 0.8
    train_dataset, test_dataset = (
        dataset[: int(train_test_split * len(dataset))],
        dataset[int(train_test_split * len(dataset)) :],
    )
    pdb.set_trace()
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    class GCN(nn.Module):
        def __init__(self):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(379, 64)
            self.conv2 = GCNConv(64, 64)
            self.fc = nn.Linear(64, 1)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = torch.relu(x)
            x = self.conv2(x, edge_index)
            x = torch.relu(x)

            x = global_mean_pool(x, data.batch)

            x = self.fc(x)

            return x

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

    net_params = {
        "in_channels": 379,
        "hidden_channels": 64,
        "out_channels": 1,
        "num_layers": 2,
        "dropout": 0.5,
        "readout": "mean",
    }

    class GIN_pyg(nn.Module):
        def __init__(self, net_params):
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

            self.conv1 = GCNConv(self.in_channels, self.hidden_channels)
            self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)
            self.fc = nn.Linear(self.hidden_channels, 1)

        def forward(self, data):
            x, edge_index, edge_weight, batch = (
                data.x,
                data.edge_index,
                data.edge_weight.unsqueeze(-1),
                data.batch,
            )
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.conv2(x, edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = global_mean_pool(x, batch)
            predict = self.fc(x)

            return predict

    model = GCN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        for i, data in enumerate(train_dataloader):
            y_pred = model(data)
            loss = criterion(y_pred, data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

        correct = 0
        total = 0
        y_true = []
        y_preds = []
        for i, data in enumerate(test_dataloader):
            y_pred = model(data)
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.detach().numpy()
            y_true.extend(data.y)
            y_preds.extend(y_pred)
        y_true = np.array(y_true)
        y_preds = np.concatenate(y_preds)
        # print()
        # acc = accuracy_score(y_true, y_preds)
        auc = roc_auc_score(y_true, y_preds)
        print(f"Epoch {epoch},auc: {auc}")
