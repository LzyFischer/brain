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

from dgl.data import DGLDataset

from typing import Any, Dict, List, Tuple

# set the path to the root directory of the project
sys.path.append(os.path.abspath("../"))

DATA_PATH = os.path.abspath("./dataset/data_dict_FC.pkl")


def pre_transform(data: Dict[str, Any]) -> Data:
    """Transform the data into torch Data type"""
    x = torch.tensor(data["FC"], dtype=torch.float32)
    # normalize the data
    x = (x - x.mean()) / x.std()
    x_SC = torch.tensor(data["SC"], dtype=torch.float32)
    x_SC = (x_SC - x_SC.mean()) / x_SC.std()

    edge_index_FC = torch.tensor(
        np.stack((x > 1).nonzero()), dtype=torch.long
    ).t().contiguous()
    row, col = edge_index_FC
    edge_weight_FC = torch.tensor(x[row, col], dtype=torch.float32)
    edge_index_SC = torch.tensor(
        np.stack((x_SC > 1).nonzero()), dtype=torch.long
    ).t().contiguous()
    row, col = edge_index_SC
    edge_weight_SC = torch.tensor(x_SC[row, col], dtype=torch.float32)

    feature = torch.tensor(data['feature'], dtype=torch.float32).unsqueeze(0) if 'feature' in data.keys() else None
    label_tensor = torch.tensor(data['label'], dtype=torch.float32).unsqueeze(0)
    return Data(
        x=x,
        x_SC=x_SC,
        edge_index_FC=edge_index_FC,
        edge_weight_FC=edge_weight_FC,
        edge_index_SC=edge_index_SC,
        edge_weight_SC=edge_weight_SC,
        y=label_tensor,
        feature=feature
    )


class Brain(InMemoryDataset):
    def __init__(
        self,
        task,
        x_attributes,
        processed_path="./data/processed",
        rawdata_path=DATA_PATH,
        suffix=None,
    ):
        
        assert task in [
            "classification",
            "regression",
        ], "Task should be either 'classification' or 'regression'"

        if suffix is None:
            suffix = ""
        self.processed_path = os.path.join(processed_path, f"{task}_data{suffix}.pt")

        self.task = task
        self.x_attributes = x_attributes
        self.rawdata_path = rawdata_path
        self.suffix = suffix
        self.pre_transform = pre_transform
        # self.dir_path

        super().__init__(pre_transform=self.pre_transform)

        self.data, self.slices = torch.load(self.processed_path)

        # remove -1 missing values
        self.data.y = torch.where(self.data.y < 0, torch.tensor([0.0]), self.data.y)

        """modify"""
        self.data.y = (self.data.y)[:,[4]]
        # self.data.y = torch.where(self.data.y > 0, torch.tensor([1.0]), torch.tensor([0.0]))
        """modify end"""

        # down sample the data to make the dataset balanced
        

        # y = torch.where(self.data.y == -1, torch.tensor([0.0]), self.data.y)
        
        # for i in range(len(self.data.y[0])):
        #     missing_value = torch.where(self.data.y == -1, torch.tensor([1.0]), torch.tensor([0.0]))[:,i].sum()
        #     print(f"Missing value in {i}th sample: {missing_value}, positive label: {y[:,i].sum()}, total label: {y[:,i].shape[0]}"
        #     )
        # pdb.set_trace()

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
    # buckets = {"Age in year": [(8, 12), (12, 18), (18, 23)]}
    # attributes_group = ["Age in year"]
    # dataset1 = HCDP( task='classification',
    #                x_attributes=['FC', 'SC', 'CT'], y_attribute='age',
    #                buckets_dict=buckets, attributes_to_group=attributes_group)
    # print(dataset1[0], len(dataset1))
    # for i in range(200):
    #     print(dataset1[i])
    dataset2 = Brain(
        task="classification",
        x_attributes=["FC", "SC"],
    )
    print(len(dataset2))
