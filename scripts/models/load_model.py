# -*- coding: utf-8 -*-
# @Time    : 10/3/2023 8:57 PM
# @Author  : Gang Qu
# @FileName: load_model.py
from models.GIN_pyg import GIN_pyg
from models.MLP_pyg import MLP_pyg
from models.AutoEncoder import AE
import pdb

def load_model(config, args):
    if config['model'] == 'GIN_pyg':
        model = GIN_pyg(config['net_params'], args)
    elif config['model'] == 'MLP_pyg':
        model = MLP_pyg(config['net_params'], args)
    elif config['model'] == 'AE':
        model = AE(config['net_params'], args)
    elif config['model'] == 'Linear':
        model = MLP(config['net_params'])
    return model
