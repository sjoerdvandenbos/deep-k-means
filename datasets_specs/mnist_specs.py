#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

from pathlib import Path
from utils import read_list
from sklearn.datasets import fetch_openml
from utils import ImgSet
import numpy as np

data_path = Path.cwd() / "split" / "mnist"
print(f"(Down)loading dataset...")
if (data_path / "compacted_data.npy").exists():
    data = np.load(data_path / "compacted_data.npy")
    target = np.load(data_path / "compacted_target.npy")
else:
    # Fetch the dataset
    download = fetch_openml("mnist_784", version=1, cache=True)
    data = download.data.to_numpy()
    target = download.target.to_numpy().astype(np.int64)
print("Dataset MNIST loaded...")
img_height = 28
img_width = 28
n_samples = data.shape[0]   # Number of samples in the dataset
n_clusters = 10             # Number of clusters to obtain
n_channels = 1
data = data.reshape(n_samples, n_channels, img_height, img_width)

# Get the split between training/test set and validation set
train_indices = read_list(data_path / "train")
test_indices = read_list(data_path / "validation")

trainset = ImgSet(data[train_indices], target[train_indices])
testset = ImgSet(data[test_indices], target[test_indices])

# Auto-encoder architecture
input_size = img_height * img_width
