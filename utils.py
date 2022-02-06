#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

from pathlib import Path

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
    truth_map = get_clusterlabel_to_groundtruth_map(gtruth, cluster_label)
    return np.array([truth_map[e.item()] for e in cluster_label])


def get_clusterlabel_to_groundtruth_map(gtruths, cluster_labels):
    """ Returns a map clabel -> gtruth. """
    gtruths = gtruths.long()
    cluster_labels = cluster_labels.long()
    D = max(cluster_labels.max(), gtruths.max()) + 1
    w = torch.zeros((D, D), dtype=torch.long)
    for i in range(cluster_labels.size(0)):
        w[cluster_labels[i], gtruths[i]] += 1
    ind = linear_assignment(w.max() - w) # Optimal label mapping based on the Hungarian algorithm
    return dict(zip(ind[0], ind[1]))


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


def load_dataset(path):
    print("loading dataset...")
    data = np.load(path / "compacted_data.npy")
    target = np.load(path / "compacted_target.npy")
    train_indices = read_list(path / "train")
    validation_indices = read_list(path / "validation")
    print("done loading")
    return data, target, train_indices, validation_indices


def write_dataset(path, name, data, target, train_indices, validation_indices):
    print("writing dataset...")
    new_dir = path / name
    new_dir.mkdir()
    np.save(new_dir / "compacted_data.npy", data)
    np.save(new_dir / "compacted_target.npy", target)
    write_list(new_dir / "train", train_indices)
    write_list(new_dir / "validation", validation_indices)
    print("done writing")


def read_list(file_name, type='int'):
    with open(file_name, 'r') as f:
        lines = f.read().splitlines()
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


def get_color_map(n_colors, is_darker=False):
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(i/n_colors) for i in range(n_colors)]
    if is_darker:
        colors = [_darken_rgba(c) for c in colors]
    return dict(enumerate(colors))


def _darken_rgba(rgba_tuple, factor=1.5):
    r, g, b, a = rgba_tuple
    return r/factor, g/factor, b/factor, a


def path_contains_dataset(path):
    path_obj = Path(__file__).parent / "split" / path
    files = path_obj.glob("*")
    names = [f.name for f in files]
    data_exists = (
            "compacted_data.npy" in names
            and "compacted_target.npy" in names
            and "train" in names
            and "validation" in names
    )
    if data_exists:
        return True
    else:
        print(f"no (complete) dataset found at given path: {path}\n files found here: {names}")
        return False
