from pathlib import Path

import numpy as np
import pandas as pd
import skimage.measure

from ptb_img_utils import get_filenames_and_labels
from utils import read_list


def get_matrix_data(dataset_dir):
    filenames, diseases = get_filenames_and_labels(dataset_dir)
    print("loading matrix data", flush=True)
    data = []
    for f in filenames:
        matrix = pd.read_csv(f, sep=",", header=None, nrows=15, low_memory=False, dtype=np.float64, engine="c").to_numpy()
        reduced = skimage.measure.block_reduce(matrix, (1, 7), np.max, cval=-3)
        data.append(reduced.flatten())
    data = np.asarray(data)
    print("matrix data loaded", flush=True)
    return data, diseases


def get_matrix_data_from_npy(dataset_dir, target_file):
    print("loading matrix data", flush=True)
    data = np.load(dataset_dir / "compacted_data.npy")
    if target_file[-3:] == "csv":
        diseases = read_list(dataset_dir / target_file, "str")
    else:
        diseases = np.load(dataset_dir / target_file).flatten()
    print("done", flush=True)
    return data, diseases


def save_matrix_as_numpy(matrix, diseases, dataset_dir):
    print("writing data to disk as .npy...", flush=True)
    np.save(dataset_dir / "compacted_data.npy", matrix)
    np.save(dataset_dir / "compacted_target.npy", diseases)
    print("done", flush=True)


if __name__ == "__main__":
    dataset_path = Path.cwd() / "split" / "ptb-matrices"
    print(dataset_path)
    data, diseases = get_matrix_data(dataset_path)
    save_matrix_as_numpy(data, diseases, dataset_path)
