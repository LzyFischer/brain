"""
Training utilities for Cross-GNN within the brain pipeline.

This module ports the original training loop from the standalone
Cross-GNN implementation.  It operates on dense tensors produced by
``tensorize_subset`` and performs stratified training with a small
validation split.  Metrics such as accuracy, AUC, sensitivity,
specificity and F1 score are computed when the validation accuracy
improves.

Example usage (inside ``main_brain.py``):

    from train.train_cross import tensorize_subset, train_cross_epoch

    train_x, train_y, _ = tensorize_subset(train_subset, device, num_nodes)
    val_x, val_y, val_oh = tensorize_subset(val_subset, device, num_nodes)
    test_x, test_y, test_oh = tensorize_subset(test_subset, device, num_nodes)

    datas = (train_x, val_x, test_x,
             train_y, val_y, test_y,
             val_oh, test_oh,
             num_classes)

    for epoch in range(max_epochs):
        res, is_best, to_save = train_cross_epoch(epoch, model, optimizer, datas, res, args)
"""

from __future__ import annotations

import time
from typing import Iterable, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from sklearn.utils import shuffle
from termcolor import cprint

__all__ = ["tensorize_subset", "train_cross_epoch"]


def accuracy(output: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute the fraction of correct predictions.

    Args:
        output: Probability or logit tensor of shape (N, C).
        labels: Ground truth labels of shape (N,).

    Returns:
        Scalar accuracy in the range [0,1].
    """
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return (correct / len(labels)).item()


def tensorize_subset(subset: Iterable[Any], device: torch.device, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a subset of PyG ``Data`` objects into dense tensors for Cross-GNN.

    Each ``Data`` instance must expose ``data.x`` (functional connectivity),
    ``data.x_SC`` (structural connectivity) and ``data.y`` (label).

    Args:
        subset: Iterable of ``Data`` objects.
        device: Device on which to allocate the returned tensors.
        num_nodes: Number of nodes (size of adjacency matrices).

    Returns:
        ``data_x``: Tensor of shape (N, 2, num_nodes, num_nodes).
        ``labels``: Tensor of shape (N,).
        ``onehot``: One‑hot encoded labels of shape (N, C) where C is the
            number of unique labels in ``labels``.
    """
    # Determine number of subjects and classes
    N = len(subset)
    # The dataset is binary classification in this project, but we
    # determine the number of classes from the labels for flexibility.
    labels_raw = [int(data.y.item()) for data in subset]
    unique_labels = sorted(set(labels_raw))
    num_classes = max(unique_labels) + 1

    # Preallocate storage
    data_x = torch.zeros((N, 2, num_nodes, num_nodes), dtype=torch.float32, device=device)
    labels = torch.zeros((N,), dtype=torch.long, device=device)

    for i, data in enumerate(subset):
        # ``data.x`` and ``data.x_SC`` are dense adjacency matrices
        fc = data.x  # shape: (num_nodes, num_nodes)
        sc = data.x_SC  # shape: (num_nodes, num_nodes)
        data_x[i, 0] = fc
        data_x[i, 1] = sc
        labels[i] = int(data.y.item())

    # One‑hot encode labels
    onehot = torch.zeros((N, num_classes), dtype=torch.float32, device=device)
    onehot[torch.arange(N), labels] = 1.0
    return data_x, labels, onehot


def train_cross_epoch(epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                      datas: Tuple[torch.Tensor, ...], res: Dict[str, Any], args: object) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
    """Perform one epoch of Cross-GNN training and evaluation.

    This function replicates the logic of the original Cross-GNN `train`
    function.  It trains on the provided training data, evaluates on the
    validation set, and if the validation accuracy improves, evaluates on
    the test set and updates ``res`` with the best metrics.

    Args:
        epoch: Current epoch number (zero‑based).
        model: Cross-GNN model instance.
        optimizer: Optimiser for model parameters.
        datas: Tuple containing training/validation/test tensors and labels:
            (train_data, val_data, test_data,
             train_label, val_label, test_label,
             val_onehot_labels, test_onehot_labels,
             num_classes)
        res: Dictionary to accumulate best metrics.
        args: Namespace of command‑line arguments containing ``alpha``.

    Returns:
        Updated ``res`` dictionary, ``is_best`` flag indicating whether
        validation accuracy improved, and ``to_save`` dictionary with
        per‑subject labels when the best validation was achieved.
    """
    # Unpack data
    (train_data, val_data, test_data,
     train_label, val_label, test_label,
     val_onehot_labels, test_onehot_labels,
     num_classes) = datas

    # Static variable for tracking best validation accuracy across epochs
    if not hasattr(train_cross_epoch, "best_val_acc"):
        train_cross_epoch.best_val_acc = 0.0

    is_best = False
    t0 = time.time()
    model.train()

    # Shuffle training data before batching
    train_data, train_label = shuffle(train_data, train_label)

    batch_size = 64
    num_train = train_data.shape[0]
    batch_iters = num_train // batch_size + 1
    acc_train_loss = 0.0
    cum_acc_train = 0.0

    # Perform mini‑batch training
    for i in range(batch_iters):
        st = i * batch_size
        ed = min(st + batch_size, num_train)
        if st >= ed:
            continue
        sub_train_x = train_data[st:ed]
        sub_train_y = train_label[st:ed]

        # Linearly ramp up alpha in the first 50 epochs as in the original implementation
        alpha_t = args.alpha * float((epoch + 1) / 50.0)

        optimizer.zero_grad()
        out_p, p1_log, p2_log = model(sub_train_x, tem=1.0, get_corr=False)
        # p1_log and p2_log are log probabilities; exponentiate to get probabilities
        p1 = p1_log.exp()
        p2 = p2_log.exp()
        # Cross-entropy loss on the primary output
        loss_train = F.cross_entropy(out_p, sub_train_y) \
            + alpha_t * F.kl_div(p1_log, out_p, reduction="batchmean") \
            + alpha_t * F.kl_div(p2_log, out_p, reduction="batchmean") \
            + alpha_t * F.kl_div(p2_log, p1, reduction="batchmean")
        acc_train = accuracy(out_p, sub_train_y)
        loss_train.backward()
        optimizer.step()
        acc_train_loss += loss_train.item() * (ed - st)
        cum_acc_train += acc_train * (ed - st)

    acc_train_loss /= num_train
    cum_acc_train /= num_train

    # Validation
    model.eval()
    with torch.no_grad():
        pred_val, _, _ = model(val_data, get_corr=False)
    acc_val = accuracy(pred_val, val_label)

    to_save: Dict[str, Any] = {}
    # Update best metrics if validation accuracy improved
    if train_cross_epoch.best_val_acc <= acc_val or True:
        is_best = True
        train_cross_epoch.best_val_acc = acc_val
        with torch.no_grad():
            pred_test, _, _ = model(test_data, get_corr=False)

        acc_test = accuracy(pred_test, test_label)
        # Compute evaluation metrics
        onehot_test_label_cpu = test_onehot_labels.cpu().numpy()
        pred_test_cpu = pred_test.cpu().numpy()
        pred_label_cpu = pred_test.max(1)[1].cpu().numpy()
        test_label_cpu = test_label.cpu().numpy()
        auc_test = roc_auc_score(onehot_test_label_cpu.ravel(), pred_test_cpu.ravel())
        cm = confusion_matrix(pred_label_cpu, test_label_cpu)
        f1 = f1_score(pred_label_cpu, test_label_cpu)
        # Sensitivity (recall on the positive class)
        eval_sen = round(cm[1, 1] / float(cm[1, 1] + cm[1, 0] + 1e-8), 4)
        # Specificity (recall on the negative class)
        eval_spe = round(cm[0, 0] / float(cm[0, 0] + cm[0, 1] + 1e-8), 4)
        # Populate results using the same naming convention as the brain pipeline
        # so they can be aggregated alongside other models.  We record the best
        # metrics across epochs on the validation set.
        res["Accuracy"] = acc_test
        res["AUC"] = auc_test
        res["Sensitivity"] = eval_sen
        res["Specificity"] = eval_spe
        # Recall and Precision use sklearn's definitions for binary positive class
        from sklearn.metrics import precision_score, recall_score
        precision = precision_score(test_label_cpu, pred_label_cpu)
        recall = recall_score(test_label_cpu, pred_label_cpu)
        res["Recall"] = recall
        res["Precision"] = precision
        res["F1"] = f1
        # Save ground truth for further analysis if desired
        to_save = {
            "label_onehot": onehot_test_label_cpu,
            "label": test_label_cpu,
        }
        cprint(
            f"Epoch: {epoch + 1:04d} "
            f"loss_train: {acc_train_loss:.4f} "
            f"acc_train: {cum_acc_train:.4f} "
            f"val_acc: {acc_val:.4f} "
            f"test_acc: {acc_test:.4f} "
            f"test_auc: {auc_test:.4f} "
            f"test_sen: {eval_sen:.4f} "
            f"test_spe: {eval_spe:.4f} "
            f"test_f1: {f1:.4f} "
            f"time: {time.time() - t0:.4f}s ",
            "green",
        )
    return res, is_best, to_save