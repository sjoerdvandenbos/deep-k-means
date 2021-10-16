#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

import numpy as np
import os
from scipy.optimize import linear_sum_assignment as linear_assignment
import tensorflow as tf
import torch

TF_FLOAT_TYPE = tf.float32


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed.
    (Taken from https://github.com/XifengGuo/IDEC-toy/blob/master/DEC.py)
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    assert y_pred.size(0) == y_true.size(0)
    D = max(y_pred.max().item(), y_true.max().item()) + 1
    w = torch.zeros((D, D), dtype=torch.long)
    for i in range(y_pred.size(0)):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w) # Optimal label mapping based on the Hungarian algorithm

    # ind is a tuple of arrays, we want an array of tuples
    indices = np.array([(ind[0][i], ind[1][i]) for i in range(len(ind[0]))])
    return sum([w[i, j] for i, j in indices]) * 1.0 / y_pred.size(0)


def map_clusterlabels_to_groundtruth(clabel, gtruth):
    """ Returns a map clabel -> gtruth. """
    D = max(clabel.max().item(), gtruth.max().item()) + 1
    w = torch.zeros((D, D), dtype=torch.long)
    for i in range(clabel.size(0)):
        w[clabel[i], gtruth[i]] += 1
    ind = linear_assignment(w.max() - w) # Optimal label mapping based on the Hungarian algorithm
    return dict(zip(ind))


def next_batch(num, data):
    """
    Return a total of `num` random samples.
    """
    indices = np.arange(0, data.shape[0])
    np.random.shuffle(indices)
    indices = indices[:num]
    batch_data = np.asarray([data[i, :] / 255 for i in indices])

    return indices, batch_data


def shuffle(data, target):
    """
    Return a random permutation of the data.
    """
    indices = np.arange(0, len(data))
    np.random.shuffle(indices)
    shuffled_data = np.asarray([data[i] for i in indices])
    shuffled_labels = np.asarray([target[i] for i in indices])

    return shuffled_data, shuffled_labels, indices


def read_list(file_name, type='int'):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    if type == 'str':
        array = np.asarray([l.strip() for l in lines])
        return array
    elif type == 'int':
        array = np.asarray([int(l.strip()) for l in lines])
        return array
    else:
        print("Unknown type")
        return None


def write_list(file_name, array):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as f:
        for item in array:
          f.write("{}\n".format(item))
