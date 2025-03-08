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
from models.model_loss import weighted_mse_loss, classification_loss
from sklearn.metrics import roc_curve, auc, roc_auc_score, recall_score, precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
import pdb
from sklearn.manifold import SpectralEmbedding

# surpress warnings UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore")


LOGGING_DIC = {
    'version': 1.0,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format':
                '%(asctime)s %(threadName)s:%(thread)d [%(name)s] %(levelname)s [%(pathname)s:%(lineno)d] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'simple': {
            'format': '%(asctime)s [%(name)s] %(levelname)s %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'test': {
            'format': '%(asctime)s %(message)s',
        },
    },
    'filters': {},
    'handlers': {
        'console_debug_handler': {
            'level': 'DEBUG',  # 日志处理的级别限制
            'class': 'logging.StreamHandler',  # 输出到终端
            'formatter': 'simple'  # 日志格式
        },
        'file_info_handler': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',  # 保存到文件,日志轮转
            'filename': 'user.log',
            'maxBytes': 1024 * 1024 * 10,  # 日志大小 10M
            'backupCount': 10,  # 日志文件保存数量限制
            'encoding': 'utf-8',
            'formatter': 'standard',
        },
        'file_debug_handler': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',  # 保存到文件
            'filename': 'test.log',  # 日志存放的路径
            'encoding': 'utf-8',  # 日志文件的编码
            'formatter': 'test',
        },
    },
    'loggers': {
        'logger1': {  # 导入时logging.getLogger时使用的app_name
            'handlers': ['console_debug_handler'],  # 日志分配到哪个handlers中
            'level': 'DEBUG',
            'propagate': False,
        },
        'logger2': {
            'handlers': ['console_debug_handler', 'file_debug_handler'],
            'level': 'INFO',
            'propagate': False,
        },
    }
}






def get_logger(name, path='results/loggers'):
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
    log_config['handlers']['file_debug_handler']['filename'] = os.path.join(path, name)
    logging.config.dictConfig(log_config)
    logger = logging.getLogger('logger2')
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



def log_metrics(epoch_loss, predicted_scores, target_scores, task_type, phase, epoch, writer, logger, multi_label=True):
    if task_type == "regression":
        rmse = evaluate_mat(predicted_scores, target_scores, method='RMSE')
        mae = evaluate_mat(predicted_scores, target_scores, method='MAE')
        log_or_print('{} loss={} RMSE={} MAE={}'.format(phase, epoch_loss, rmse, mae), logger)
        if writer:
            writer.add_scalar('Loss/' + phase, epoch_loss, epoch)
            for idx, (r, m) in enumerate(zip(rmse, mae)):
                writer.add_scalar('Metrics/RMSE_{}_{}'.format(idx, phase), r, epoch)
                writer.add_scalar('Metrics/MAE_{}_{}'.format(idx, phase), m, epoch)
    elif task_type == "classification":
        # Convert predicted scores to class label
        predicted_scores = np.concatenate(predicted_scores, axis=0)
        target_scores = np.concatenate(target_scores, axis=0)
        threshold = 0.5

        n_classes = predicted_scores.shape[1]
        if n_classes <=2:
            accuracy = accuracy_score(target_scores, (predicted_scores > threshold).astype(float))
            recall = recall_score(target_scores, (predicted_scores > threshold).astype(float))
            precision = precision_score(target_scores, (predicted_scores > threshold).astype(float))
            f1 = f1_score(target_scores, (predicted_scores > threshold).astype(float))
            auc = roc_auc_score(target_scores, predicted_scores)
            log_or_print('{} Loss: {:.4f}, Accuracy: {:.2f}%, Recall: {:.4f}, Precision: {:.4f}, F1: {:.4f}, AUC: {:.4f}'.format(phase, epoch_loss, accuracy, recall, precision, f1, auc), logger)
            if writer:
                writer.add_scalar('Loss/' + phase, epoch_loss, epoch)
                writer.add_scalar('Metrics/Accuracy_' + phase, accuracy, epoch)
                writer.add_scalar('Metrics/Recall_' + phase, recall, epoch)
                writer.add_scalar('Metrics/Precision_' + phase, precision, epoch)
                writer.add_scalar('Metrics/F1_' + phase, f1, epoch)
                writer.add_scalar('Metrics/AUC_' + phase, auc, epoch)
        else:
            # calculate the tpr, tnr, fpr, fnr for each class
            accuracy_all = []
            recall_all = []
            precision_all = []
            f1_all = []
            auc_all = []
            for i in range(predicted_scores.shape[1]):
                accuracy_all.append(accuracy_score(target_scores[:, i], (predicted_scores[:, i] > threshold).astype(float)))
                recall_all.append(recall_score(target_scores[:, i], (predicted_scores[:, i] > threshold).astype(float)))
                precision_all.append(precision_score(target_scores[:, i], (predicted_scores[:, i] > threshold).astype(float)))
                f1_all.append(f1_score(target_scores[:, i], (predicted_scores[:, i] > threshold).astype(float)))
                auc_all.append(roc_auc_score(target_scores[:, i], predicted_scores[:, i]))
                log_or_print('{} Loss: {:.4f}, Accuracy: {:.2f}%, Recall: {:.4f}, Precision: {:.4f}, F1: {:.4f}, AUC: {:.4f}'.format(phase, epoch_loss, accuracy_all[i], recall_all[i], precision_all[i], f1_all[i], auc_all[i]), logger)
            accuracy = np.mean(accuracy_all)
            recall = np.mean(recall_all)
            precision = np.mean(precision_all)
            f1 = np.mean(f1_all)
            auc = np.mean(auc_all)
            log_or_print('{} Loss: {:.4f}, Accuracy_Average: {:.2f}%, Recall_Average: {:.4f}, Precision_Average: {:.4f}, F1_Average: {:.4f}, AUC_average: {:.4f}'.format(phase, epoch_loss, accuracy, recall, precision, f1, auc), logger)



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
            data = [torch.tensor(d).to(dtype=torch.float32, device='cuda') if isinstance(d, np.ndarray) else d.to(
                dtype=torch.float32, device='cuda') for d in data]
            data = torch.cat(data, dim=0)
        elif isinstance(data, np.ndarray):
            data = torch.tensor(data).to(dtype=torch.float32, device='cuda')
        # Ensure the resulting tensor is on the right device and dtype
        return data.to(dtype=torch.float32, device='cuda')

    # Convert lists of tensors to single tensors if needed
    predicted = to_single_tensor(predicted)
    target = to_single_tensor(target)

    # Calculate the difference between the predicted and target scores.
    res = predicted - target

    if method == 'RMSE':
        res = res ** 2
        # Assuming res is a PyTorch tensor for efficient computation
        res = torch.sqrt(torch.sum(res, dim=0) / res.shape[0]).cpu().numpy()

    elif method == 'MAE':
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


class EarlyStopping():
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