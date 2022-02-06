from pathlib import Path

import numpy as np

from utils import read_list, ImgSet
from ptb_matrix_utils import get_matrix_data_from_npy


class PTBSpecs:
    def __init__(self, path):
        self.dataset_path = Path(__file__).parent.parent / "split" / path
        self.data, self.diseases = get_matrix_data_from_npy(self.dataset_path)
        print(self.data.shape)
        print(self.diseases.shape)
        self.inverse_disease_mapping = dict(enumerate(np.unique(self.diseases)))
        self.disease_mapping = {v: k for k, v in self.inverse_disease_mapping.items()}
        self.target = np.fromiter((self.disease_mapping[d] for d in self.diseases), dtype=int)
        self.n_clusters = len(self.inverse_disease_mapping)
        self.n_samples, self.n_channels, self.img_height, self.img_width = self.data.shape
        # Get the split between training/test set and validation set
        test_indices = read_list(self.dataset_path / "validation")
        train_indices = read_list(self.dataset_path / "train")
        self.trainset = ImgSet(self.data[train_indices], self.target[train_indices])
        self.testset = ImgSet(self.data[test_indices], self.target[test_indices])
