#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

import numpy as np
import os
from scipy.optimize import linear_sum_assignment as linear_assignment
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


class ImgSet(Dataset):

    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.data = torch.as_tensor(source)
        self.target = torch.as_tensor(target)

    def __getitem__(self, index):
        transformed = (self.data[index] / 255).float()
        return index, transformed, self.target[index]

    def __len__(self):
        return self.data.shape[0]


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
    correct = np.zeros_like(y_true)
    for i in range(correct.shape[0]):
        correct[i] = 1 if y_pred[i] == y_true[i] else 0
    return sum(correct) / correct.shape[0]


def map_clusterlabels_to_groundtruth(gtruth, cluster_label):
    """ Returns a map clabel -> gtruth. """
    D = max(cluster_label.max().item(), gtruth.max().item()) + 1
    w = torch.zeros((D, D), dtype=torch.long)
    for i in range(cluster_label.size(0)):
        w[cluster_label[i], gtruth[i]] += 1
    ind = linear_assignment(w.max() - w) # Optimal label mapping based on the Hungarian algorithm
    truth_map = dict(zip(ind[0], ind[1]))
    return np.array([truth_map[e.item()] for e in cluster_label])


def conv_regularization_loss(autoencoder):
    conv_layers = _get_conv_layers(autoencoder)
    loss = 0
    for conv in conv_layers:
        loss += _conv_regularization(conv.weight, conv.stride)
    return loss


def _get_conv_layers(autoencoder):
    all_modules = [*autoencoder.encoder, *autoencoder.decoder]
    only_convs = [
        m for m in all_modules
        if type(m) == torch.nn.Conv2d
        or type(m) == torch.nn.ConvTranspose2d
    ]
    return only_convs


def _conv_regularization(kernel, stride=2, padding=1):
    """ Copied from https://github.com/samaonline/Orthogonal-Convolutional-Neural-Networks/blob
    /aa7f56901c661a124e0cfe72eb2c9dc98045ce94/imagenet/utils.py#L34"""
    out_channels = kernel.shape[0]
    self_conv = torch.conv2d(kernel, kernel, stride=stride, padding=padding)
    target = torch.zeros((out_channels, out_channels, self_conv.shape[-2], self_conv.shape[-1])).cuda()
    ct = int(np.floor(self_conv.shape[-1]/2))
    target[:, :, ct, ct] = torch.eye(out_channels).cuda()
    return torch.norm(self_conv - target)


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


def get_color_map(n_colors):
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(i/n_colors) for i in range(n_colors)]
    return dict(enumerate(colors))
