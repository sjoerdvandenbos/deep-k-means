from typing import Dict

import numpy as np
from pathlib import Path

from utils import ImgSet
from torch import randperm


data_path = Path.cwd() / "split" / "mit-plots"
print("Loading MIT image set...")
data = np.load(data_path / "compacted_data.npy").astype(np.uint8)
diseases = np.load(data_path / "compacted_target.npy", allow_pickle=True).astype(np.str).flatten()
INVERTED_DISEASE_MAPPING: Dict[int, str] = dict(enumerate(np.unique(diseases)))
DISEASE_MAPPING: Dict[str, int] = {v: k for k, v in INVERTED_DISEASE_MAPPING.items()}
target = np.fromiter((DISEASE_MAPPING[d] for d in diseases), dtype=np.int32)
print("Done loading data")
num_channels = 1
n_samples = target.shape[0]
img_height = data.shape[1]
img_width = data.shape[2]
n_clusters = len(DISEASE_MAPPING)
data = data.reshape((n_samples, 1, img_height, img_width))      # data format: [N, C, H, W]

# Get the split between training/test set and validation set
indices = randperm(n_samples)
percolation_boundary = int(n_samples * 0.7)
train_indices = indices[:percolation_boundary]
test_indices = indices[percolation_boundary:]
trainset = ImgSet(data[train_indices], target[train_indices])
testset = ImgSet(data[test_indices], target[test_indices])
