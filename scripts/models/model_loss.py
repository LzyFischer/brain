# -*- coding: utf-8 -*-
# @Time    : 10/3/2023 9:06 PM
# @Author  : Gang Qu
# @FileName: model_loss.py
import torch.nn.functional as F
import torch
import pdb

def weighted_mse_loss(output, targets, weights=None):
    """
    Weighted Mean Squared Error (MSE) for multivariate regression task.
    Args:
        output: Predicted output from the model.
        target: Ground truth.
        weights: Optional tensor of weights for each variable.
    Returns:
        Weighted loss value.
    """
    mask = (targets != -1).float()
    masked_targets = targets * mask
    if weights is None:
        loss = F.mse_loss(output, masked_targets, reduction='none')
        loss = loss * mask
        loss = loss / mask.sum()
        return loss

    squared_diffs = (output - masked_targets) ** 2

    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=squared_diffs.device).view(1, -1)
    weighted_squared_diffs = weights_tensor * squared_diffs

    weighted_squared_diffs = weighted_squared_diffs * mask
    weighted_squared_diffs = weighted_squared_diffs / mask.sum()

    return weighted_squared_diffs


def classification_loss(output, target, args):
    """
    Multi-label binary cross-entropy loss for multi-label classification task with mask.

    Args:
        output (torch.Tensor): Predicted output from the model (logits).
        target (torch.Tensor): Ground truth labels (-1 for masked, 0/1 for labels).
        args (Namespace): Contains arguments like loss_type and focal_gamma.

    Returns:
        torch.Tensor: Mean loss value.
    """
    loss = []
    if args.loss_type == 'multi_class':
        target = target.squeeze()
        mask = (target != -1).float()  # Create mask for valid targets
        target_masked = target.masked_fill(target == -1, 0).long() # fill -1 with 0 and convert to long for cross entropy.
        ce_loss = F.cross_entropy(output, target_masked, reduction='none')
        masked_ce_loss = ce_loss * mask
        loss = masked_ce_loss.mean()
        return loss

    for i in range(output.size(1)):  # Iterate through each label
        mask = (target[:, i] != -1).float()  # Create mask for valid labels
        target_masked = target[:, i].masked_fill(target[:, i] == -1, 0) # fill -1 with 0, so bce loss can work.

        if args.loss_type == 'bce':
            bce = F.binary_cross_entropy_with_logits(output[:, i], target_masked, reduction='none')
            loss.append((bce * mask).mean())

        elif args.loss_type == 'focal':
            loss.append(focal_loss(output[:, i], target_masked, args.focal_gamma, mask))

        elif args.loss_type == 'weighted_bce':
            pos_samples = (target_masked * mask).sum()
            neg_samples = mask.sum() - pos_samples
            pos_weight = neg_samples / pos_samples if pos_samples > 0 else torch.tensor(0.0)

            bce = F.binary_cross_entropy_with_logits(output[:, i], target_masked, pos_weight=pos_weight, reduction='none')
            loss.append((bce * mask).mean())

        else:
            raise ValueError("Invalid loss type. Choose from 'bce', 'focal', or 'weighted_bce'.")

    for i in range(len(loss)):
        if torch.isnan(loss[i]):
            loss[i] = torch.tensor(0.0, requires_grad=True)

    return torch.stack(loss).mean()


def focal_loss(predictions, targets, alpha=0.25, gamma=2.0, mask=None):
    """Focal loss with optional mask."""
    bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss

    if mask is not None:
        focal_loss = focal_loss * mask

    return focal_loss.mean()

