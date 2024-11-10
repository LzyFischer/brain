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

def get_bucket(value, buckets):
    """Return the bucket/range for a given value."""
    for start, end in buckets:
        if start <= value < end:
            return f"{start}-{end}"
    return "Other"


def safe_float_conversion(value):
    try:
        return float(value)
    except ValueError:
        return value


def group_by_attributes(data, buckets_dict, attributes_to_group):
    """
    Group subjects based on specified buckets and attributes.

    Parameters:
    - data (dict): Dictionary containing subject IDs as keys and another dictionary of attributes as values.
    - buckets_dict (dict): Dictionary specifying buckets for each attribute.
    - attributes_to_group (list): List of attributes to group by.

    Returns:
    - dict: A dictionary where keys are attribute values or combinations, and values are lists of subject IDs.
    """

    def safe_float_conversion(value):
        try:
            return float(value)
        except ValueError:
            return value

    grouped = {}

    for subject, attributes in data.items():
        keys = []
        skip_subject = False
        for attr in sorted(attributes_to_group):  # sort the attributes_to_group for consistent order
            # Skip this subject if it's missing one of the attributes_to_group
            if attr not in attributes or attributes[attr] is None:
                skip_subject = True
                break

            value = safe_float_conversion(attributes[attr])

            if attr in buckets_dict:
                keys.append(get_bucket(value, sorted(buckets_dict[attr])))  # ensure the buckets are sorted
            else:
                keys.append(value)

        if skip_subject:
            continue

        key = tuple(keys) if len(keys) > 1 else keys[0]

        if key not in grouped:
            grouped[key] = []
        grouped[key].append(subject)

    return grouped


def get_labels_for_subjects_and_attributes(grouped_data, buckets_dict):
    labels_for_subjects = []
    attribute_for_labels = {}
    current_label = 0

    # Go over each attribute in buckets_dict in the defined order
    for attribute, bucket_ranges in buckets_dict.items():
        for bucket_range in bucket_ranges:
            group = f"{bucket_range[0]}-{bucket_range[1]}"
            if group in grouped_data:
                attribute_for_labels[current_label] = group
                subjects = grouped_data[group]
                for subject in subjects:
                    labels_for_subjects.append((subject, current_label))
                current_label += 1

    # Now handle any additional groups in grouped_data that aren't in buckets_dict
    for group, subjects in grouped_data.items():
        if group not in attribute_for_labels.values():  # To avoid re-processing
            attribute_for_labels[current_label] = group
            for subject in subjects:
                labels_for_subjects.append((subject, current_label))
            current_label += 1

    return labels_for_subjects, attribute_for_labels


def get_connection(x, k=None, threshold=None, binarize=False, is_symmetric=True):
    """
    Process the matrix x to get an adjacency matrix.
    """

    def keep_k_largest_for_rows(mat, k):
        # We sort values in ascending order and then keep only the largest k for each row.
        # The rest are set to 0.
        sorted_indices = np.argsort(mat, axis=1)
        row_indices = np.arange(mat.shape[0])[:, None]
        mat[row_indices, sorted_indices[:, :-k]] = 0
        return mat

    is_tensor = isinstance(x, torch.Tensor)
    if is_tensor:
        x = x.cpu().numpy()

    def process_matrix(matrix):
        if k:
            matrix = keep_k_largest_for_rows(matrix, k)

        if threshold:
            matrix[matrix < threshold] = 0

        if binarize:
            matrix = np.where(matrix > 0, 1, 0)

        if is_symmetric:
            matrix = np.maximum(matrix, matrix.T)

        return matrix

    if len(x.shape) == 3:
        for idx in range(x.shape[0]):
            x[idx] = process_matrix(x[idx])
    else:
        x = process_matrix(x)

    if is_tensor:
        x = torch.from_numpy(x)

    return x


