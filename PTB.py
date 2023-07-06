from pathlib import Path

import numpy as np

from utils import read_list, ImgSet, MatrixSet
from ptb_matrix_utils import get_matrix_data_from_npy


class PTBSpecs:
    def __init__(self, path, target_file, data_format):
        self.target_file = target_file
        self.dataset_path = Path(__file__).parent / "split" / path
        self.data, self.diseases = get_matrix_data_from_npy(self.dataset_path, target_file)
        print(self.data.shape)
        print(self.diseases.shape)
        self.inverse_disease_mapping = dict(enumerate(np.unique(self.diseases)))
        self.disease_mapping = {v: k for k, v in self.inverse_disease_mapping.items()}
        self.target = np.fromiter((self.disease_mapping[d] for d in self.diseases), dtype=int)
        self.n_clusters = 2     # len(self.inverse_disease_mapping)
        self.n_samples, self.n_channels, self.img_height, self.img_width = self.data.shape
        self.indices_are_present = self._dataset_dir_contains("validation") and self._dataset_dir_contains("train")
        # Get the split between training/test set and validation set
        if self.indices_are_present:
            test_indices = read_list(self.dataset_path / "validation")
            train_indices = read_list(self.dataset_path / "train")
        else:
            train_indices, test_indices = self._split_indices(train_fraction=0.7)
        self.train_indices = train_indices
        self.test_indices = test_indices
        if data_format == "IMAGE" or data_format == "IMAGE_FIELDS":
            self.trainset = ImgSet(self.data[train_indices], self.target[train_indices])
            self.testset = ImgSet(self.data[test_indices], self.target[test_indices])
        else:
            self.trainset = MatrixSet(self.data[train_indices], self.target[train_indices])
            self.testset = MatrixSet(self.data[test_indices], self.target[test_indices])
        patient_ids = read_list(self.dataset_path / "patient_numbers.csv")
        self.train_ids = patient_ids[train_indices]
        self.test_ids = patient_ids[test_indices]

    def _dataset_dir_contains(self, filename):
        path_objects = self.dataset_path.glob("*")
        file_names = [path.name for path in path_objects]
        return filename in file_names

    def _split_indices(self, train_fraction):
        label_names = np.unique(self.diseases)
        names_to_indices = {name: [] for name in label_names}
        # Fill names_to_indices
        for i, name in enumerate(self.diseases):
            names_to_indices[name].append(i)
        train_indices = []
        test_indices = []
        # Make the same random split for every label
        for name in label_names:
            indices = np.array(names_to_indices[name])
            np.random.shuffle(indices)
            first_test_index = int(len(indices) * train_fraction)
            train_indices.extend(indices[:first_test_index])
            test_indices.extend(indices[first_test_index:])
        return np.array(train_indices), np.array(test_indices)

