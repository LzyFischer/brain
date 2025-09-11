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
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import pdb

def train_epoch(model, optimizer, device, data_loader, epoch, task_type="regression", logger=None, writer=None,
                weight_score=None, alpha=None, beta=None, args=None):
    model.train()
    model = model.to(device)
    
    # initialize merics
    epoch_loss = 0
    predicted_scores = []
    target_scores = []

    # log info
    utils.log_or_print('-' * 60, logger)
    utils.log_or_print('EPOCH:{} (lr={})'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']), logger)

    # training begin
    for iter, sample_i in enumerate((data_loader)):
        optimizer.zero_grad()
        sample_i = sample_i.to(device)
        outputs = model(sample_i)
        # if is tuple
        if isinstance(outputs, tuple):
            batch_scores, extra_loss, att_FCs, att_SCs = outputs
        else:
            batch_scores = outputs
            extra_loss = 0
            att_FCs = None
            att_SCs = None

        # Compute the loss
        if task_type == "regression":
            loss = weighted_mse_loss(batch_scores, sample_i.y, weight_score)
        elif task_type == "classification":
            loss = classification_loss(batch_scores, sample_i.y, args=args)
            if torch.isnan(loss):
                loss = torch.tensor(0.0, requires_grad=True)
        else:
            raise ValueError("Invalid task type. Choose from 'regression' or 'classification'.")
        
        loss += extra_loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()

        # note here we use sigmoid!!!
        batch_scores = F.sigmoid(batch_scores) if task_type == "classification" else batch_scores
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

    att_FCs_list = None
    att_SCs_list = None
    with torch.no_grad():
        for iter, sample_i in enumerate(data_loader):
            outputs = model(sample_i.to(device))
            # if is tuple
            if isinstance(outputs, tuple):
                batch_scores, extra_loss, att_FCs, att_SCs = outputs
                if epoch == args.max_epochs - 1:
                    # mask: ground truth == 1
                    mask = (sample_i.y.view(-1) == 1)   # (B,)

                    if mask.any():
                        att_FCs_sel = att_FCs[mask]
                        att_SCs_sel = att_SCs[mask]

                        if att_FCs_list is None:  # 初始化
                            att_FCs_list = att_FCs_sel
                            att_SCs_list = att_SCs_sel
                        else:  # 拼接
                            att_FCs_list = torch.cat((att_FCs_list, att_FCs_sel), dim=0)
                            att_SCs_list = torch.cat((att_SCs_list, att_SCs_sel), dim=0)
            else:
                batch_scores = outputs
                extra_loss = 0
                att_FCs = None
                att_SCs = None

            if task_type == "regression":
                loss = weighted_mse_loss(batch_scores, sample_i.y, weight_score)
            elif task_type == "classification":
                loss = classification_loss(batch_scores, sample_i.y, args=args)
            else:
                raise ValueError("Invalid task type. Choose from 'regression' or 'classification'.")

            epoch_test_loss += loss.detach().item()
            batch_scores = F.sigmoid(batch_scores)
            predicted_scores.append(batch_scores.cpu().detach().numpy())
            target_scores.append(sample_i.y.cpu().detach().numpy())
    
    epoch_test_loss /= (iter + 1)
    metrics = utils.log_metrics(epoch_test_loss, predicted_scores, target_scores, task_type, phase, epoch, writer, logger)

    if att_FCs_list is not None and epoch == args.max_epochs-1:
        # save the attention weights
        np.savez(f"results/attention_weights/{args.data_name}.npz", 
                 att_FCs=att_FCs_list.cpu().numpy(), att_SCs=att_SCs_list.cpu().numpy())
        G_FC = np.stack([model.layers[i].apf_FC.Wg.cpu().detach().numpy() for i in range(len(model.layers))], axis=1)
        G_SC = np.stack([model.layers[i].apf_SC.Wg.cpu().detach().numpy() for i in range(len(model.layers))], axis=1)
        np.savez(f"results/attention_weights/{args.data_name}_G.npz", G_FC=G_FC, G_SC=G_SC)

    return epoch_test_loss, metrics



def train_epoch_anomaly_detection(model, optimizer, device, data_loader, epoch, task_type="regression", logger=None, phase='Val', writer=None,
                     weight_score=None, args=None):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(data_loader):
        # Ensure batch contains only normal samples (positive class)
        if hasattr(batch, "y"):
            normal_samples = batch.y == 0  # Assuming 0 is the label for normal class
            if normal_samples.sum() == 0:
                continue  # Skip if no normal samples in the batch
            # get the normal samplee index
            index = torch.where(batch.y == 0)[0]
            batch = Subset(batch, index)
            # subset to databatch
            batch = Batch.from_data_list(batch)
        else:
            raise AttributeError("The input data must have a 'y' attribute for anomaly detection tasks.")

        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch)

        # Compute reconstruction loss
        target = batch.x_SC  # Target is the same as input for reconstruction
        target = global_mean_pool(target, batch.batch)  # Global pooling for node-level input
        loss = F.mse_loss(outputs, target)  # Example loss (can be customized)

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Logging for every N batches
        if batch_idx % 10 == 0:
            logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(data_loader)
    logger.info(f"Epoch {epoch}, Training Loss: {avg_loss:.4f}")
    writer.add_scalar("Train/Loss", avg_loss, epoch)

    return avg_loss, optimizer



def evaluate_network_anomaly_detection(model, device, data_loader, epoch, task_type="regression", logger=None, phase='Val', writer=None,
                     weight_score=None, args=None):
    model.eval()
    total_loss = 0
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)

            # Forward pass
            outputs = model(batch)

            # Compute reconstruction loss
            target = batch.x_SC
            target = global_mean_pool(target, batch.batch)
            loss = F.mse_loss(outputs, target, reduction="none")  # Compute loss per sample
            loss = loss.mean(dim=1)  # Aggregate per sample

            # Collect ground-truth labels and reconstruction scores
            all_labels.append(batch.y.cpu().numpy())
            all_scores.append(loss.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)

    # Classification based on threshold
    threshold = np.percentile(all_scores, 95)
    predictions = (all_scores > threshold).astype(int)

    # calculate roc_auc, normalize the score
    all_scores = (all_scores - all_scores.min()) / (all_scores.max() - all_scores.min())
    auc = roc_auc_score(all_labels, all_scores)

    # Compute evaluation metrics
    precision = precision_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    accuracy = np.mean(all_labels == predictions)

    logger.info(f"Epoch {epoch}, {phase} Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    writer.add_scalar(f"{phase}/Precision", precision, epoch)
    writer.add_scalar(f"{phase}/Recall", recall, epoch)
    writer.add_scalar(f"{phase}/F1", f1, epoch)

    return -precision