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
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
import pdb

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

        # Convert predicted scores to class label
        # predicted_labels = np.argmax(predicted_scores, axis=1)
        predicted_labels = np.where(predicted_scores > 0.5, 1, 0).squeeze()
        true_labels = np.concatenate(target_scores)
        correct_predictions = np.sum(predicted_labels == true_labels, axis=0)

        # Calculate the accuracy
        # accuracy = correct_predictions / len(true_labels) * 100
        # accuracy = accuracy_score(true_labels.astype(float), predicted_labels.astype(float))
        accuracy = correct_predictions.mean() / len(true_labels) * 100
        # Assuming there are n_classes
        n_classes = predicted_scores.shape[1]
        # Compute ROC and AUC for either binary or multiclass classification
        if n_classes <= 2:
            fpr, tpr, _ = roc_curve(true_labels, predicted_scores[:, 0])
            roc_auc = auc(fpr, tpr)
            # For Tensorboard: log the curve points
            if writer:
                writer.add_scalar('Metrics/AUC_' + phase, roc_auc, epoch)

                for i in range(len(fpr)):
                    writer.add_scalars('ROC_curve_' + phase,{'False Positive Rate': fpr[i], 'True Positive Rate': tpr[i]}, i)
        else:
            if multi_label:
                # Compute ROC and AUC for multilabel classification
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(n_classes):
                    # pdb.set_trace()
                    if true_labels[:, i].sum() == 0:
                        continue
                    try:
                        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], predicted_scores[:, i])
                    except:
                        pdb.set_trace()
                    if not (np.isnan(fpr[i]).any() or np.isnan(tpr[i]).any()):
                        roc_auc[i] = auc(fpr[i], tpr[i])
                        # For Tensorboard: log the curve points for each class
                        if writer:
                            writer.add_scalar('Metrics/AUC_class_' + str(i) + "_" + phase, roc_auc[i], epoch)
                            for j in range(len(fpr[i])):
                                writer.add_scalars('ROC_curve_class_' + str(i) + "_" + phase,
                                                {'False Positive Rate': fpr[i][j], 'True Positive Rate': tpr[i][j]}, j)

                # # Log the AUC values
                # for i in roc_auc.keys():
                #     log_or_print(f"{phase} Loss: {epoch_loss:.4f}, AUC for class {i} = {roc_auc[i]:.4f}, Accuracy: {accuracy:.2f}%",
                #                  logger)
            else:
                # Binarize the labels for one-vs-all ROC computation
                true_labels_bin = label_binarize(true_labels, classes=np.arange(n_classes))
                false_positive_rate = dict()
                true_positive_rate = dict()
                roc_auc = dict()
                for i in range(n_classes):
                    false_positive_rate[i], true_positive_rate[i], _ = roc_curve(true_labels_bin[:, i],
                                                                                predicted_scores[:, i])
                    roc_auc[i] = auc(false_positive_rate[i], true_positive_rate[i])

                    # For Tensorboard: log the curve points for each class
                    if writer:
                        writer.add_scalar('Metrics/AUC_class_' + str(i) + "_" + phase, roc_auc[i], epoch)
                        for j in range(len(false_positive_rate[i])):
                            writer.add_scalars('ROC_curve_class_' + str(i) + "_" + phase,
                                            {'False Positive Rate': false_positive_rate[i][j],
                                                'True Positive Rate': true_positive_rate[i][j]}, j)

        # log the AUC values
        if n_classes <= 2:
            log_or_print(f"{phase} Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, AUC={roc_auc:.4f}", logger)
        else:
            # for i in roc_auc.keys():
            #     log_or_print(f"{phase} Loss: {epoch_loss:.4f}, AUC for class {i} = {roc_auc[i]:.4f}, Accuracy: {accuracy:.2f}%",
            #                  logger)
            roc_auc = np.mean(list(roc_auc.values()))
            log_or_print(f"{phase} Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, AUC={roc_auc:.4f}", logger)


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

