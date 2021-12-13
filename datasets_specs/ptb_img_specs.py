import numpy as np
from pathlib import Path

from utils import read_list, ImgSet


data_path = Path.cwd() / "split" / "ptb-concatted-leads" / "3k_per_disease_3_diseases"
print("Loading PTB image set...")
data = np.load(data_path / "compacted_data.npy").astype(np.uint8)
diseases = np.load(data_path / "compacted_target.npy").astype(np.str).flatten()
inverse_disease_mapping = dict(enumerate(np.unique(diseases)))
disease_mapping = {v: k for k, v in inverse_disease_mapping.items()}
target = np.fromiter((disease_mapping[d] for d in diseases), dtype=np.int32)
print("Done loading data")

data = data.reshape((9600, 1, 314, 384))

n_samples = target.shape[0]
print(f"data shape: {data.shape}")
n_channels = data.shape[1]
img_height = data.shape[2]
img_width = data.shape[3]
n_clusters = len(disease_mapping)
print(f"number of classes: {n_clusters}")

# Get the split between training/test set and validation set
train_indices = read_list(data_path / "train")
test_indices = read_list(data_path / "validation")

trainset = ImgSet(data[train_indices], target[train_indices])
testset = ImgSet(data[test_indices], target[test_indices])

input_size = img_height * img_width
