# -*- coding: utf-8 -*-
# @Time    : 10/3/2023 9:03 PM
# @Author  : Gang Qu
# @FileName: train.py
import torch
import torch.nn as nn
import dgl
import numpy as np
from utils import utils
from tqdm import tqdm
from models.model_loss import weighted_mse_loss, classification_loss
from models.MASKGCN import SymmetricMaskedGraphConv

import pdb

def train_epoch(model, optimizer, device, data_loader, epoch, task_type="regression", logger=None, writer=None,
                weight_score=None, alpha=None, beta=None, args=None):
    model.train()
    model = model.to(device)
    epoch_loss = 0
    predicted_scores = []
    target_scores = []

    utils.log_or_print('-' * 60, logger)
    utils.log_or_print('EPOCH:{} (lr={})'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']), logger)

    for iter, sample_i in enumerate(tqdm(data_loader, desc="Training iterations!")):
        optimizer.zero_grad()
        sample_i = sample_i.to(device)
        batch_scores = model(sample_i)
        if task_type == "regression":
            loss = weighted_mse_loss(batch_scores, sample_i.y, weight_score)
        elif task_type == "classification":
            loss = classification_loss(batch_scores, sample_i.y)

        else:
            raise ValueError("Invalid task type. Choose from 'regression' or 'classification'.")
        mask_weights = model.layers[0].raw_edge_weight if isinstance(model.layers[0], SymmetricMaskedGraphConv) else None
        if mask_weights is not None:
            l1_reg = torch.norm(mask_weights, p=1)
            l2_reg = torch.norm(mask_weights, p=2)

            # Add the regularization terms to the primary loss
            loss += alpha * l1_reg + beta * l2_reg
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        predicted_scores.append(batch_scores.cpu().detach().numpy())
        target_scores.append(sample_i.y.cpu().detach().numpy())


    epoch_loss /= (iter + 1)
    utils.log_metrics(epoch_loss, predicted_scores, target_scores, task_type, 'Train', epoch, writer, logger)

    return epoch_loss, optimizer


def evaluate_network(model, device, data_loader, epoch, task_type="regression", logger=None, phase='Val', writer=None,
                     weight_score=None, args=None):
    model.eval()
    epoch_test_loss = 0
    predicted_scores = []
    target_scores = []

    with torch.no_grad():
        for iter, sample_i in enumerate(tqdm(data_loader, desc=phase + " iterations!")):
            batch_scores = model(sample_i.to(device))

            if task_type == "regression":
                loss = weighted_mse_loss(batch_scores, sample_i.y, weight_score)
            elif task_type == "classification":
                loss = classification_loss(batch_scores, sample_i.y)
            else:
                raise ValueError("Invalid task type. Choose from 'regression' or 'classification'.")

            epoch_test_loss += loss.detach().item()
            predicted_scores.append(batch_scores.cpu().detach().numpy())
            target_scores.append(sample_i.y.cpu().detach().numpy())

    epoch_test_loss /= (iter + 1)
    utils.log_metrics(epoch_test_loss, predicted_scores, target_scores, task_type, phase, epoch, writer, logger)

    return epoch_test_loss



if __name__ == "__main__":
    import json
    from scripts.datasets.HCDP_dataset import HCDP, HCDP_DGL
    from scripts.models.GIN import GIN

    with open(r'F:\projects\AiyingT1MultimodalFusion\scripts\configs\GIN.json') as f:
        config = json.load(f)
    model = GIN(config['net_params'])
    print(model)

    buckets = {"Age in year": [(8, 12), (12, 18), (18, 23)]}
    attributes_group = ["Age in year"]
    # dataset1 = HCDP( task='classification',
    #                x_attributes=['SC', 'FC'], y_attribute='age',
    #                buckets_dict=buckets, attributes_to_group=attributes_group)
    # print(dataset1[0], len(dataset1))
    #
    # dataset2 = HCDP( task='regression',
    #                x_attributes=['SC', 'FC'], y_attribute=['nih_crycogcomp_ageadjusted', 'nih_fluidcogcomp_ageadjusted'],
    #                buckets_dict=buckets, attributes_to_group=attributes_group)
    # print(dataset2[0], len(dataset2))

    dataset3 = HCDP_DGL(task='regression',
                        x_attributes=['FC', 'SC', 'CT'],
                        y_attribute=['nih_crycogcomp_ageadjusted', 'nih_fluidcogcomp_ageadjusted'],
                        buckets_dict=buckets, attributes_to_group=attributes_group)

    print(dataset3[0], len(dataset3))

    # for sub, gi, ys in dataset3:
    #     y = ys[0]
    #     print(gi.ndata['x'][:, :360], gi.ndata['x'][:, 360:])
    #     break
    #     y_, g = model(gi, gi.ndata['x'][:, :360])
    #     print(y_, y_.shape)
    from torch.utils.data import DataLoader
    from scripts.utils.utils import collate_dgl
    train_dataloader = DataLoader(dataset3, batch_size=64, shuffle=True, collate_fn=collate_dgl)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for e in range(10):
        epoch_loss, optimizer, gs = train_epoch(model, optimizer, device=torch.device('cuda'), data_loader=train_dataloader, epoch=e, task_type="regression", logger=None, writer=None,
                weight_score=None)
        print(epoch_loss)