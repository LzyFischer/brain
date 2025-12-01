# -*- coding: utf-8 -*-
# @Time    : 10/3/2023 8:57 PM
# @Author  : Gang Qu
# @FileName: load_model.py
from models.GCN_pyg import GCN_pyg
from models.MLP_pyg import MLP_pyg
from models.Transformers import BrainTransformers
from models.AutoEncoder import AE
from models.CrossGNN import CrossGNNBrain
import pdb

def load_model(config, args):
    if config['model'] == 'GCN_pyg':
        model = GCN_pyg(config['net_params'], args)
    elif config['model'] == 'MLP_pyg':
        model = MLP_pyg(config['net_params'], args)
    elif config['model'] == 'AE':
        model = AE(config['net_params'], args)
    elif config['model'] == 'Linear':
        model = MLP(config['net_params'])
    elif config['model'] == 'BrainTransformers':
        model = BrainTransformers(config['net_params'], args)
    elif config['model'] == 'CrossGNNBrain':
        # Wrap the Cross-GNN architecture so it can consume PyG Data batches.
        # ``net_params`` should at minimum contain ``in_channels`` and
        # optionally ``num_classes``.
        model = CrossGNNBrain(config['net_params'], args)
    return model
