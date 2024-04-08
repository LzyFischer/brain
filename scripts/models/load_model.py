# -*- coding: utf-8 -*-
# @Time    : 10/3/2023 8:57 PM
# @Author  : Gang Qu
# @FileName: load_model.py
from models.GCN import GCN
from models.GIN import GIN
from models.GAT import GAT
from models.MLP import MLP
from models.MASKGCN import MASKGCN
def load_model(config):
    if config['model'] == 'GCN':
        model = GCN(config['net_params'])
    elif config['model'] == 'GIN':
        model = GIN(config['net_params'])
    elif config['model'] == 'GAT':
        model = GAT(config['net_params'])
    elif config['model'] == 'MLP':
        model = MLP(config['net_params'])
    elif config['model'] == 'Linear':
        model = MLP(config['net_params'])
    elif config['model'] == 'MASKGCN':
        model = MASKGCN(config['net_params'])
    return model
