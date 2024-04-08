# -*- coding: utf-8 -*-
# @Time    : 10/3/2023 9:06 PM
# @Author  : Gang Qu
# @FileName: model_loss.py
import torch.nn.functional as F
import torch


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
    Cross Entropy Loss for classification task.
    Args:
        output: Predicted output from the model.
        target: Ground truth.
    Returns:
        Loss value.
    """
    return F.cross_entropy(output, target)

