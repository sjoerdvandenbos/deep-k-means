import numpy as np
import pandas as pd

from pathlib import Path

import torch

from dataset_summary import summarize
from utils import read_list, write_list, load_dataset, write_dataset

import torch.nn.functional as F


def shrink_dataset_by_number_and_filter(train_size, test_size, labels_to_use, dataset_dir, name):
    old_train_indices, old_test_indices, target = _get_old_indices_and_target(dataset_dir)
    filtered_train_indices = _filter_indices_on_label(labels_to_use, old_train_indices, target)
    filtered_test_indices = _filter_indices_on_label(labels_to_use, old_test_indices, target)
    resized_train_indices = _get_resized_indices(train_size, filtered_train_indices, target)
    resized_test_indices = _get_resized_indices(test_size, filtered_test_indices, target)
    _write_smaller_npy(dataset_dir, resized_train_indices, resized_test_indices, name)


def shrink_single_dataset(train_size, test_size, dataset_dir, name):
    old_train_indices, old_test_indices, target = _get_old_indices_and_target(dataset_dir)
    new_train_indices = _get_resized_indices(train_size, old_train_indices, target)
    new_test_indices = _get_resized_indices(test_size, old_test_indices, target)
    _write_smaller_npy(dataset_dir, new_train_indices, new_test_indices, name)


def shrink_dataset_by_label_filtering(labels_to_use, dataset_dir, name):
    old_train_indices, old_test_indices, target = _get_old_indices_and_target(dataset_dir)
    new_train_indices = _filter_indices_on_label(labels_to_use, old_train_indices, target)
    new_test_indices = _filter_indices_on_label(labels_to_use, old_test_indices, target)
    _write_smaller_npy(dataset_dir, new_train_indices, new_test_indices, name)


def shrink_matrices_dataset_by_leads(dataset_dir, leads=np.arange(12)):
    print("Shrinking by reducing number of leads...")
    train_indices = read_list(dataset_dir / "train")
    test_indices = read_list(dataset_dir / "validation")
    target = np.load(dataset_dir / "compacted_target.npy")
    data = np.load(dataset_dir / "compacted_data.npy")
    new_data = data[:, :, leads, :]
    destination_dir = dataset_dir / f"{len(leads)}leads"
    destination_dir.mkdir()
    np.save(destination_dir / "compacted_data.npy", new_data)
    np.save(destination_dir / "compacted_target.npy", target)
    write_list(destination_dir / "train", train_indices)
    write_list(destination_dir / "validation", test_indices)
    summarize(destination_dir)
    print("Done")


def shrink_matrices_dataset_by_avgpool(dataset_dir, poolsize, name):
    data, target, train_indices, validation_indices = load_dataset(dataset_dir)
    tensor_data = torch.from_numpy(data)
    pooled_tensor = F.avg_pool2d(tensor_data, (1, poolsize))
    pooled_numpy = pooled_tensor.detach().float().numpy()
    write_dataset(dataset_dir, name, pooled_numpy, target, train_indices, validation_indices)


def _get_old_indices_and_target(dataset_path):
    old_train_indices = pd.read_csv(dataset_path / "train", header=None) \
        .to_numpy().flatten()
    old_test_indices = pd.read_csv(dataset_path / "validation", header=None) \
        .to_numpy().flatten()
    target = np.load(dataset_path / "compacted_target.npy").flatten()
    return old_train_indices, old_test_indices, target


def _filter_indices_on_label(labels_to_use, old_indices, target):
    new_indices = np.fromiter((i for i in old_indices if target[i] in labels_to_use), dtype=int)
    return new_indices


def _get_resized_indices(new_size, old_indices, target):
    disease_names = np.unique(target)
    names_to_indices = {name: [] for name in disease_names}
    # Fill names_to_indices
    for i in old_indices:
        names_to_indices[target[i]].append(i)
    # Pick new_size indices per disease randomly
    for name in disease_names:
        indices = np.array(names_to_indices[name])
        np.random.shuffle(indices)
        names_to_indices[name] = indices[:new_size]
    # Collect all indices in a single list
    resized_indices = []
    for index_list in names_to_indices.values():
        resized_indices.extend(index_list)
    return np.array(resized_indices)


def _write_smaller_npy(path, train_indices, test_indices, name):
    data, target = _load_and_merge_npy(path, train_indices, test_indices)
    destination = path / name
    destination.mkdir(parents=False, exist_ok=True)
    save_npy(destination, data, target)
    # Write train indices
    pd.DataFrame(np.arange(0, len(train_indices))).to_csv(destination / "train", index=False, header=False)
    # Write test indices
    first_test_index = len(train_indices)
    last_test_index = len(train_indices) + len(test_indices)
    pd.DataFrame(np.arange(first_test_index, last_test_index)).to_csv(destination / "validation", index=False,
                                                                      header=False)
    summarize(destination)


def _load_and_merge_npy(path, train_indices, test_indices):
    data, target = load_npy(path)
    train_data = data[train_indices]
    train_target = target[train_indices]
    test_data = data[test_indices]
    test_target = target[test_indices]
    return (
        np.concatenate((train_data, test_data), axis=0),
        np.concatenate((train_target, test_target), axis=0),
    )


def load_npy(path):
    data = np.load(path / "compacted_data.npy")
    target = np.load(path / "compacted_target.npy")
    return data, target


def save_npy(path, data, target):
    np.save(path / "compacted_data", data)
    np.save(path / "compacted_target", target)


if __name__ == "__main__":
    np.random.seed(500)
    directory = Path(__file__).parent / "split" / "ptb-12lead-matrices-40hz"
    labels = ("HealthyControl", "BundleBranchBlock")
    # shrink_matrices_dataset_by_avgpool(directory, 25, "ptb-12lead-matrices-40hz")
    # shrink_matrices_dataset_by_leads(
    #     dataset_dir="ptb-matrices",
    #     leads=np.arange(12),
    # )
    shrink_dataset_by_label_filtering(
        labels_to_use=labels,
        dataset_dir=directory,
        name=f"all_samples_{len(labels)}_diseases"
    )
    # shrink_dataset_by_number_and_filter(
    #     train_size=3000,
    #     test_size=600,
    #     labels_to_use=("HealthyControl", "BundleBranchBlock", "Myocarditis"),
    #     dataset_dir=directory,
    #     name="3k_per_disease_3_diseases"
    # )
