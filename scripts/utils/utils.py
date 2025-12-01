# -*- coding: utf-8 -*-
# @Time    : 8/16/2022 10:42 AM
# @Author  : Gang Qu
# @FileName: utils.py
import os
import torch
import numpy as np
import copy
import logging.config
import random
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from scipy import sparse as sp
import dgl
import torch.nn.functional as F
from models.model_loss import classification_loss
from sklearn.metrics import roc_curve, auc, roc_auc_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
)
from sklearn.utils import resample
from torch_geometric.data import Dataset as PyGDataset
import pdb
from sklearn.manifold import SpectralEmbedding

from torch_geometric.data import Dataset
from sklearn.utils import resample
from matplotlib.colors import ListedColormap
from matplotlib import colors

# surpress warnings UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore")


LOGGING_DIC = {
    "version": 1.0,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s %(threadName)s:%(thread)d [%(name)s] %(levelname)s [%(pathname)s:%(lineno)d] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {
            "format": "%(asctime)s [%(name)s] %(levelname)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "test": {
            "format": "%(asctime)s %(message)s",
        },
    },
    "filters": {},
    "handlers": {
        "console_debug_handler": {
            "level": "DEBUG",  # 日志处理的级别限制
            "class": "logging.StreamHandler",  # 输出到终端
            "formatter": "simple",  # 日志格式
        },
        "file_info_handler": {
            "level": "INFO",
            "class": "logging.handlers.RotatingFileHandler",  # 保存到文件,日志轮转
            "filename": "user.log",
            "maxBytes": 1024 * 1024 * 10,  # 日志大小 10M
            "backupCount": 10,  # 日志文件保存数量限制
            "encoding": "utf-8",
            "formatter": "standard",
        },
        "file_debug_handler": {
            "level": "DEBUG",
            "class": "logging.FileHandler",  # 保存到文件
            "filename": "test.log",  # 日志存放的路径
            "encoding": "utf-8",  # 日志文件的编码
            "formatter": "test",
        },
    },
    "loggers": {
        "logger1": {  # 导入时logging.getLogger时使用的app_name
            "handlers": ["console_debug_handler"],  # 日志分配到哪个handlers中
            "level": "DEBUG",
            "propagate": False,
        },
        "logger2": {
            "handlers": ["console_debug_handler", "file_debug_handler"],
            "level": "INFO",
            "propagate": False,
        },
    },
}


def get_logger(name, path="results/loggers"):
    """
    get the logger
    :param name: name of logger file
    :type name:
    :return:
    :rtype:
    """
    log_config = copy.deepcopy(LOGGING_DIC)
    if not os.path.exists(path):
        os.makedirs(path)
    log_config["handlers"]["file_debug_handler"]["filename"] = os.path.join(path, name)
    logging.config.dictConfig(log_config)
    logger = logging.getLogger("logger2")
    return logger


def seed_it(seed):
    """
    set random seed for reproducibility
    :param seed: seed number
    :type seed: int
    :return:
    :rtype:
    """
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def _safe_confusion_binary(y_true, y_pred_bin):
    """Return tn, fp, fn, tp for binary labels; robust if a class is missing."""
    con = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
    if con.shape != (2, 2):
        full = np.zeros((2, 2), dtype=int)
        for i in range(con.shape[0]):
            for j in range(con.shape[1]):
                full[i, j] = con[i, j]
        con = full
    tn, fp, fn, tp = con.ravel()
    return tn, fp, fn, tp

def compute_sensitivity_specificity(y_true, y_pred_bin):
    """Compute (sensitivity, specificity) for binary predictions."""
    tn, fp, fn, tp = _safe_confusion_binary(y_true, y_pred_bin)
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return sen, spe


def log_metrics(
    epoch_loss,
    predicted_scores,
    target_scores,
    task_type,
    phase,
    epoch,
    writer,
    logger,
    multi_label=True,
):
    """
    Extended: now also logs Sensitivity (TPR) and Specificity (TNR).
    - Binary: single pair (sen, spe)
    - Multi-label: per-class + macro-average
    """
    if task_type == "regression":
        rmse = evaluate_mat(predicted_scores, target_scores, method="RMSE")
        mae = evaluate_mat(predicted_scores, target_scores, method="MAE")
        log_or_print(
            "{} loss={} RMSE={} MAE={}".format(phase, epoch_loss, rmse, mae), logger
        )
        if writer:
            writer.add_scalar("Loss/" + phase, epoch_loss, epoch)
            for idx, (r, m) in enumerate(zip(rmse, mae)):
                writer.add_scalar("Metrics/RMSE_{}_{}".format(idx, phase), r, epoch)
                writer.add_scalar("Metrics/MAE_{}_{}".format(idx, phase), m, epoch)
        return

    # ---------- Classification ----------
    predicted_scores = np.concatenate(predicted_scores, axis=0)
    target_scores = np.concatenate(target_scores, axis=0)
    threshold = 0.5
    n_classes = predicted_scores.shape[1]

    if n_classes <= 2:  # Binary
        mask = (target_scores != -1).astype(bool)
        target_masked = target_scores[mask]
        predicted_masked = predicted_scores[mask]

        if target_masked.size == 0:
            log_or_print(f"{phase} Loss: {epoch_loss:.4f}, All targets masked", logger)
            if writer:
                writer.add_scalar("Loss/" + phase, epoch_loss, epoch)
            return

        y_pred_bin = (predicted_masked > threshold).astype(float)

        accuracy = accuracy_score(target_masked, y_pred_bin)
        recall = recall_score(target_masked, y_pred_bin)
        precision = precision_score(target_masked, y_pred_bin)
        f1 = f1_score(target_masked, y_pred_bin)
        try:
            auc = roc_auc_score(target_masked, predicted_masked)
        except Exception:
            auc = 0.0

        # Sensitivity/Specificity
        sen, spe = compute_sensitivity_specificity(target_masked, y_pred_bin)

        log_or_print(
            "{} Loss: {:.4f}, Acc: {:.2f}%, Sen: {:.4f}, Spe: {:.4f}, Recall: {:.4f}, Precision: {:.4f}, F1: {:.4f}, AUC: {:.4f}".format(
                phase, epoch_loss, accuracy, sen, spe, recall, precision, f1, auc
            ),
            logger,
        )

        if writer:
            writer.add_scalar("Loss/" + phase, epoch_loss, epoch)
            writer.add_scalar("Metrics/Accuracy_" + phase, accuracy, epoch)
            writer.add_scalar("Metrics/Sensitivity_" + phase, sen, epoch)
            writer.add_scalar("Metrics/Specificity_" + phase, spe, epoch)
            writer.add_scalar("Metrics/Recall_" + phase, recall, epoch)
            writer.add_scalar("Metrics/Precision_" + phase, precision, epoch)
            writer.add_scalar("Metrics/F1_" + phase, f1, epoch)
            writer.add_scalar("Metrics/AUC_" + phase, auc, epoch)

        metrics = {
            "Accuracy": accuracy,
            "Sensitivity": sen,
            "Specificity": spe,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "AUC": auc,
        }
        return metrics

    else:  # Multi-label / Multi-class (one-vs-rest)
        acc_all, rec_all, pre_all, f1_all, auc_all = [], [], [], [], []
        sen_all, spe_all = [], []
        if target_scores.squeeze().ndim == 1:
            target_scores = label_binarize(target_scores, classes=np.arange(n_classes))

        for i in range(n_classes):
            mask = (target_scores[:, i] != -1).astype(bool)
            target_masked = target_scores[mask, i]
            predicted_masked = predicted_scores[mask, i]

            if target_masked.size == 0:
                log_or_print(f"{phase} Loss: {epoch_loss:.4f}, Class {i}, All targets masked", logger)
                continue

            y_pred_bin = (predicted_masked > threshold).astype(float)

            acc_all.append(accuracy_score(target_masked, y_pred_bin))
            rec_all.append(recall_score(target_masked, y_pred_bin))
            pre_all.append(precision_score(target_masked, y_pred_bin))
            f1_all.append(f1_score(target_masked, y_pred_bin))
            try:
                auc_all.append(roc_auc_score(target_masked, predicted_masked))
            except Exception:
                auc_all.append(0.0)

            # Sen/Spe
            sen_i, spe_i = compute_sensitivity_specificity(target_masked, y_pred_bin)
            sen_all.append(sen_i)
            spe_all.append(spe_i)

            log_or_print(
                "{} Loss: {:.4f}, Class {}: Acc: {:.2f}%, Sen: {:.4f}, Spe: {:.4f}, Recall: {:.4f}, Precision: {:.4f}, F1: {:.4f}, AUC: {:.4f}".format(
                    phase, epoch_loss, i, acc_all[-1], sen_i, spe_i, rec_all[-1], pre_all[-1], f1_all[-1], auc_all[-1]
                ),
                logger,
            )

        # Macro averages
        accuracy = float(np.mean(acc_all)) if acc_all else 0.0
        recall = float(np.mean(rec_all)) if rec_all else 0.0
        precision = float(np.mean(pre_all)) if pre_all else 0.0
        f1 = float(np.mean(f1_all)) if f1_all else 0.0
        auc = float(np.mean(auc_all)) if auc_all else 0.0
        sen = float(np.mean(sen_all)) if sen_all else 0.0
        spe = float(np.mean(spe_all)) if spe_all else 0.0

        log_or_print(
            "{} Loss: {:.4f}, Acc_avg: {:.2f}%, Sen_avg: {:.4f}, Spe_avg: {:.4f}, Recall_avg: {:.4f}, Precision_avg: {:.4f}, F1_avg: {:.4f}, AUC_avg: {:.4f}".format(
                phase, epoch_loss, accuracy, sen, spe, recall, precision, f1, auc
            ),
            logger,
        )

        if writer:
            writer.add_scalar("Loss/" + phase, epoch_loss, epoch)
            writer.add_scalar("Metrics/Accuracy_Average_" + phase, accuracy, epoch)
            writer.add_scalar("Metrics/Sensitivity_Average_" + phase, sen, epoch)
            writer.add_scalar("Metrics/Specificity_Average_" + phase, spe, epoch)
            writer.add_scalar("Metrics/Recall_Average_" + phase, recall, epoch)
            writer.add_scalar("Metrics/Precision_Average_" + phase, precision, epoch)
            writer.add_scalar("Metrics/F1_Average_" + phase, f1, epoch)
            writer.add_scalar("Metrics/AUC_Average_" + phase, auc, epoch)

        metrics = {
            "Accuracy_Average": accuracy,
            "Sensitivity_Average": sen,
            "Specificity_Average": spe,
            "Recall_Average": recall,
            "Precision_Average": precision,
            "F1_Average": f1,
            "AUC_Average": auc,
        }
        return metrics

def log_or_print(message, logger, use_print=True):
    if logger:
        logger.info(message)
    if use_print:
        print(message)


def evaluate_mat(predicted, target, method):
    """
    Evaluate the RMSE or MAE between predicted and target data.

    Parameters:
    - predicted (Tensor or list of Tensors): The predicted scores.
    - target (Tensor or list of Tensors): The target scores.
    - method (str): The method to use for evaluation ('RMSE' or 'MAE').

    Returns:
    numpy.ndarray: The evaluation result.
    """

    def to_single_tensor(data):
        """
        Convert data to a single PyTorch tensor.

        Parameters:
        - data: List of PyTorch tensors/NumPy ndarrays or a single tensor/ndarray

        Returns:
        - Tensor: A single concatenated PyTorch tensor
        """
        if isinstance(data, list):
            # Ensure all elements are torch tensors before concatenation
            data = [
                (
                    torch.tensor(d).to(dtype=torch.float32, device="cuda")
                    if isinstance(d, np.ndarray)
                    else d.to(dtype=torch.float32, device="cuda")
                )
                for d in data
            ]
            data = torch.cat(data, dim=0)
        elif isinstance(data, np.ndarray):
            data = torch.tensor(data).to(dtype=torch.float32, device="cuda")
        # Ensure the resulting tensor is on the right device and dtype
        return data.to(dtype=torch.float32, device="cuda")

    # Convert lists of tensors to single tensors if needed
    predicted = to_single_tensor(predicted)
    target = to_single_tensor(target)

    # Calculate the difference between the predicted and target scores.
    res = predicted - target

    if method == "RMSE":
        res = res**2
        # Assuming res is a PyTorch tensor for efficient computation
        res = torch.sqrt(torch.sum(res, dim=0) / res.shape[0]).cpu().numpy()

    elif method == "MAE":
        # Assuming res is a PyTorch tensor for efficient computation
        res = (torch.sum(torch.abs(res), dim=0) / res.shape[0]).cpu().numpy()

    else:
        raise ValueError(f"Unsupported method: {method}. Choose 'RMSE' or 'MAE'.")

    return res


def log_or_print(message, logger):
    if logger:
        logger.info(message)
    else:
        print(message)


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            self.min_delta = validation_loss - train_loss
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.counter = 0


def undersample_dataset(train_set, target_attr="y", random_state=42):
    """
    Args: train_set: torch_geometric, target_attr, random_state
    Returns: train_set
    """
    # Get all labels
    all_labels = []
    for data in train_set:
        all_labels.append(data[target_attr].item())
    all_labels = np.array(all_labels)

    # Count class frequencies
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    min_count = counts.min()

    # Get indices for each class
    balanced_indices = []
    for label in unique_labels:
        label_indices = np.where(all_labels == label)[0]
        # Undersample to the size of the smallest class
        undersampled_indices = resample(
            label_indices, replace=False, n_samples=min_count, random_state=random_state
        )
        balanced_indices.extend(undersampled_indices)

    undersampled_dataset = [train_set[i] for i in balanced_indices]

    if isinstance(train_set, Dataset):
        # If it's a custom Dataset subclass, you might need to create a new one
        # This is a simplified approach - you may need to adjust based on your specific dataset class
        from torch_geometric.data import Dataset as PyGDataset

        class UndersampledDataset(PyGDataset):
            def __init__(self, dataset_list):
                super(UndersampledDataset, self).__init__()
                self.data_list = dataset_list

            def len(self):
                return len(self.data_list)

            def get(self, idx):
                return self.data_list[idx]

        return UndersampledDataset(undersampled_dataset)


def oversample_dataset(train_set, target_attr="y", random_state=42):
    """
    Oversample minority classes in a PyTorch Geometric dataset to balance class distribution.

    Args:
        train_set: torch_geometric.data.Dataset or list of Data objects
        target_attr (str): Attribute name for the labels (default: "y")
        random_state (int): Random seed for reproducibility

    Returns:
        Oversampled dataset with balanced class distribution
    """
    # Get all labels
    all_labels = []
    for data in train_set:
        all_labels.append(data[target_attr].item())
    all_labels = np.array(all_labels)

    # Count class frequencies
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    max_count = counts.max()

    # Get indices for each class
    balanced_indices = []
    for label in unique_labels:
        label_indices = np.where(all_labels == label)[0]
        # Oversample to the size of the largest class
        oversampled_indices = resample(
            label_indices, replace=True, n_samples=max_count, random_state=random_state
        )
        balanced_indices.extend(oversampled_indices)

    oversampled_dataset = [train_set[i] for i in balanced_indices]

    if isinstance(train_set, PyGDataset):
        class OversampledDataset(PyGDataset):
            def __init__(self, dataset_list):
                super(OversampledDataset, self).__init__()
                self.data_list = dataset_list

            def len(self):
                return len(self.data_list)

            def get(self, idx):
                return self.data_list[idx]

        return OversampledDataset(oversampled_dataset)

    return oversampled_dataset

def weighted_sample(dataset, sample_type):
    if sample_type == "oversample":
        n_samples = dataset.y.shape[0]
        n_pos = (dataset.y == 1).sum().item()
        n_neg = n_samples - n_pos
        n_minority = min(n_pos, n_neg)
        n_majority = max(n_pos, n_neg)
        n_add = n_majority - n_minority
        idx_minority = torch.where(dataset.y == 1)[0]
        idx_majority = torch.where(dataset.y == 0)[0]
        idx_add = torch.randint(0, n_minority, (n_add,))
        idx_add = idx_minority[idx_add]
        data_add = dataset[idx_add]
        dataY = dataset.y
        dataset = torch.utils.data.ConcatDataset([dataset, data_add])
        dataset.y = torch.cat([dataY, data_add.y], dim=0)
    return dataset


def get_spectral_embedding(adj, d):
    adj_ = adj.data.cpu().numpy()
    emb = SpectralEmbedding(n_components=d)
    res = emb.fit_transform(adj_)
    x = torch.from_numpy(res).float().cuda()
    return x



def plot_edge_weight(edge_weight, edge_mask, fold, best_metric, save_name=None, suffix="_masked"):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --- Load and merge ROI metadata ---
    df_glasser = pd.read_excel("dataset/raw/Glasser_ROI_w_FN.xlsx")
    df_extra = pd.read_csv("dataset/raw/roi_labels.csv", header=None)
    df_extra.columns = ['LABEL']
    df_extra['NETWORK'] = 'Subcortical'
    # only last 19 rows are subcortical
    df_extra = df_extra.iloc[-19:]

    df_info = pd.concat([df_glasser[['LABEL', 'NETWORK']], df_extra], ignore_index=True)

    # --- Sort by network ---
    df_sorted = df_info.sort_values(by="NETWORK")
    labels_ordered = df_sorted["LABEL"].tolist()
    networks = df_sorted["NETWORK"].tolist()

    # --- Reorder matrices ---
    label_to_idx = {label: i for i, label in enumerate(df_info["LABEL"])}
    reorder_indices = [label_to_idx[label] for label in labels_ordered]

    edge_weight_reordered = edge_weight[reorder_indices][:, reorder_indices]
    edge_mask_reordered = edge_mask[reorder_indices][:, reorder_indices]
    ##########
    masked_weights = (edge_weight_reordered).detach().cpu().numpy()

    # # --- Plot heatmap ---
    # plt.figure(figsize=(16, 14))
    # ax = sns.heatmap(masked_weights, cmap="YlGnBu", xticklabels=False, yticklabels=False, cbar=True)
    # ax.collections[0].set_clim(0.6, 1.2)
    # # cbar fontsize
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=20)
    plt.style.use("default")  # Use default matplotlib style
    fig, ax = plt.subplots(figsize=(16, 14))
    if 1 - edge_weight.mean() > edge_weight.mean(): # closer to 1
        norm = colors.TwoSlopeNorm(vmin=0, vcenter=(masked_weights).max()/2, vmax=(masked_weights).max())
        cmap = plt.get_cmap("Blues")
    else:
        norm = colors.TwoSlopeNorm(vmin=(masked_weights).min(), vcenter=1, vmax=(masked_weights).max())
        cmap = plt.get_cmap("coolwarm")
    newcolors = cmap(np.linspace(0, 1, 256))  # upper half
    new_cmap = ListedColormap(newcolors)
    # newcmp = ListedColormap(newcolors)
    im = ax.imshow(masked_weights, interpolation="nearest", cmap=new_cmap, norm=norm)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    # no axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # borderline thickness
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    

    # --- Group boundaries ---
    group_names = []
    group_positions = []
    boundaries = []
    prev = networks[0]
    start = 0

    for i, net in enumerate(networks + ["END"]):
        if net != prev:
            group_names.append(prev)
            group_positions.append((start + i - 1) / 2)
            boundaries.append(i)
            start = i
            prev = net

    # --- Add labels ---
    for name, pos in zip(group_names, group_positions):
        plt.text(-5, pos, name, va='center', ha='right', fontsize=18)
    for name, pos in zip(group_names, group_positions):
        plt.text(pos, len(networks) + 2, name, va='top', ha='center', fontsize=18, rotation=90)

    # --- Adaptive line color ---
    mean_val = masked_weights.mean()
    line_color = 'black' if mean_val > 0.001 else 'white'
    for b in boundaries[:-1]:
        plt.axhline(b, color=line_color, linewidth=2, linestyle=(0, (6, 6)), alpha=1)
        plt.axvline(b, color=line_color, linewidth=2, linestyle=(0, (6, 6)), alpha=1)

    # --- Title & Save ---
    avg_weight = edge_weight.mean()
    plt.title(f"Fold {fold} | AUC: {best_metric:.4f} | Avg weight: {avg_weight:.8f}", fontsize=20)
    # plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.savefig(f"plot/edge_weights_fold{fold}_{suffix}.png")
    plt.close()