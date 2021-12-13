from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.datasets import CIFAR10

from utils import ImgSet, read_list

data_path = Path.cwd() / "split" / "cifar-10"
print(f"(Down)loading dataset...")
if (data_path / "compacted_data.npy").exists():
    data = np.load(data_path / "compacted_data.npy")
    target = np.load(data_path / "compacted_target.npy")
    # Get the split between training/test set and validation set
    train_indices = read_list(data_path / "train")
    test_indices = read_list(data_path / "validation")
else:
    data_path.mkdir(exist_ok=True)
    # Get data from torchvision
    train = CIFAR10(str(data_path), download=True, train=True)
    test = CIFAR10(str(data_path), download=True, train=False)
    train_imgs = torch.from_numpy(train.data).permute(0, 3, 1, 2).numpy()          # Permute changes NHWC to NCHW
    test_imgs = torch.from_numpy(test.data).permute(0, 3, 1, 2).numpy()            # Permute changes NHWC to NCHW
    train_target = train.targets
    test_target = test.targets
    data = np.concatenate((train_imgs, test_imgs)).astype(np.uint8)
    target = np.concatenate((train_target, test_target))
    train_indices = np.arange(train_imgs.shape[0])
    test_indices = np.arange(train_imgs.shape[0], train_imgs.shape[0] + test_imgs.shape[0])
    # Write to disk in fast loadable format
    np.save(data_path / "compacted_data.npy", data)
    np.save(data_path / "compacted_target.npy", target)
    pd.DataFrame(train_indices).to_csv(data_path / "train", header=False, index=False)
    pd.DataFrame(test_indices).to_csv(data_path / "validation", header=False, index=False)

print("Dataset MNIST loaded...")
img_height = data.shape[2]
img_width = data.shape[3]
n_samples = data.shape[0]                           # Number of samples in the dataset
n_clusters = np.unique(target).shape[0]             # Number of clusters to obtain
n_channels = data.shape[1]

trainset = ImgSet(data[train_indices], target[train_indices])
testset = ImgSet(data[test_indices], target[test_indices])

# Auto-encoder architecture
input_size = img_height * img_width