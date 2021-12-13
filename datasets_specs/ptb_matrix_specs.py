from pathlib import Path

import numpy as np

from utils import read_list, ImgSet
from ptb_matrix_utils import get_matrix_data_from_npy


dataset_path = Path.cwd() / "split" / "ptb-matrices" / "all_samples_3_diseases"
data, diseases = get_matrix_data_from_npy(dataset_path)
print(data.shape)
print(diseases.shape)
inverse_disease_mapping = dict(enumerate(np.unique(diseases)))
disease_mapping = {v: k for k, v in inverse_disease_mapping.items()}
target = np.fromiter((disease_mapping[d] for d in diseases), dtype=int)
n_clusters = len(inverse_disease_mapping)
n_samples, n_channels, img_height, img_width = data.shape


# Get the split between training/test set and validation set
test_indices = read_list(dataset_path / "validation")
train_indices = read_list(dataset_path / "train")

trainset = ImgSet(data[train_indices], target[train_indices])
testset = ImgSet(data[test_indices], target[test_indices])