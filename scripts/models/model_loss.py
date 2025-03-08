# -*- coding: utf-8 -*-
# @Time    : 10/3/2023 9:06 PM
# @Author  : Gang Qu
# @FileName: model_loss.py
import torch.nn.functional as F
import torch
import pdb

def focal_loss(predictions, targets, alpha=0.25, gamma=2.0):
    bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


def weighted_mse_loss(output, target, weights=None):
    """
    Weighted Mean Squared Error (MSE) for multivariate regression task.
    Args:
        output: Predicted output from the model.
        target: Ground truth.
        weights: Optional tensor of weights for each variable.
    Returns:
        Weighted loss value.
    """
    if weights is None:
        return F.mse_loss(output, target)

    squared_diffs = (output - target) ** 2

    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=squared_diffs.device).view(1, -1)
    weighted_squared_diffs = weights_tensor * squared_diffs

    return torch.mean(weighted_squared_diffs)


def classification_loss(output, target, args):
    """
    Binary cross-entropy loss for binary classification task.
    Args:
        output: Predicted output from the model, multi-label.
        target: Ground truth.
    Returns:
        Loss value.
    """
    if args.loss_type == 'bce':
        loss = []
        for i in range(output.size(1)):
            loss.append(torch.nn.BCEWithLogitsLoss()(output[:, i], target[:, i]))
    elif args.loss_type == 'focal':
        loss = []
        for i in range(output.size(1)):
            loss.append(focal_loss(output[:, i], target[:, i], args.focal_gamma))
    elif args.loss_type == 'weighted_bce':
        loss = []
        pos_samples = target.sum(dim=0)
        neg_samples = target.size(0) - pos_samples
        pos_weight = neg_samples / pos_samples
        for i in range(output.size(1)):
            loss.append(torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight[i])(output[:, i], target[:, i]))
    else:
        raise ValueError("Invalid loss type. Choose from 'bce' or 'focal'.")

    for i in range(len(loss)):
        if torch.isnan(loss[i]):
            loss[i] = torch.tensor(0.0, requires_grad=True)
    loss = torch.stack(loss).mean()
    
    return loss

