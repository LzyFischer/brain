# -*- coding: utf-8 -*-
# @Time    : 10/3/2023 9:06 PM
# @Author  : Gang Qu
# @FileName: model_loss.py
import torch.nn.functional as F
import torch
import pdb

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


def classification_loss(output, target):
    """
    Binary cross-entropy loss for binary classification task.
    Args:
        output: Predicted output from the model.
        target: Ground truth.
    Returns:
        Loss value.
    """
    # return F.binary_cross_entropy_with_logits(output, target)
    pos_weight = torch.sum(target == 0) / torch.sum(target == 1)
    loss =  torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target)
    if torch.isnan(loss):
        loss = torch.tensor(0.0, requires_grad=True)
    return loss

