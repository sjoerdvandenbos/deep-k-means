import numpy as np
import pandas as pd

from pathlib import Path


def shrink_imgs_and_matrices(train_size, test_size, name):
    """
    Takes the compacted npy files in ptb-imgs and ptb-matrices and writes shrunk down npy files in subdirectories
    in those two respective directories.
    @param train_size: the new size of the training set in SAMPLES PER DISEASE.
    @param test_size: the new size of the test set in SAMPLES PER DISEASE.
    @param name: the name of the new dataset.
    """
    split_path = Path.cwd() / "split"
    old_train_indices = pd.read_csv(split_path / "ptb-images-2-cropped" / "train", header=None)\
        .to_numpy().flatten()
    old_test_indices = pd.read_csv(split_path / "ptb-images-2-cropped" / "validation", header=None)\
        .to_numpy().flatten()
    target = np.load(split_path / "ptb-images-2-cropped" / "compacted_target.npy").flatten()
    new_train_indices = _get_resized_indices(train_size, old_train_indices, target)
    new_test_indices = _get_resized_indices(test_size, old_test_indices, target)
    _write_smaller_npy(split_path / "ptb-images-2-cropped", new_train_indices, new_test_indices, name)
    _write_smaller_npy(split_path / "ptb-matrices", new_train_indices, new_test_indices, name)


def shrink_dataset_by_number_and_filter(train_size, test_size, labels_to_use, dirname, name):
    dataset_path = Path.cwd() / "split" / dirname
    old_train_indices, old_test_indices, target = _get_old_indices_and_target(dataset_path)
    filtered_train_indices = _filter_indices_on_label(labels_to_use, old_train_indices, target)
    filtered_test_indices = _filter_indices_on_label(labels_to_use, old_test_indices, target)
    resized_train_indices = _get_resized_indices(train_size, filtered_train_indices, target)
    resized_test_indices = _get_resized_indices(test_size, filtered_test_indices, target)
    _write_smaller_npy(dataset_path, resized_train_indices, resized_test_indices, name)


def shrink_single_dataset(train_size, test_size, dirname, name):
    dataset_path = Path.cwd() / "split" / dirname
    old_train_indices, old_test_indices, target = _get_old_indices_and_target(dataset_path)
    new_train_indices = _get_resized_indices(train_size, old_train_indices, target)
    new_test_indices = _get_resized_indices(test_size, old_test_indices, target)
    _write_smaller_npy(dataset_path, new_train_indices, new_test_indices, name)


def shrink_dataset_by_label_filtering(labels_to_use, dirname, name):
    dataset_path = Path.cwd() / "split" / dirname
    old_train_indices, old_test_indices, target = _get_old_indices_and_target(dataset_path)
    new_train_indices = _filter_indices_on_label(labels_to_use, old_train_indices, target)
    new_test_indices = _filter_indices_on_label(labels_to_use, old_test_indices, target)
    _write_smaller_npy(dataset_path, new_train_indices, new_test_indices, name)


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
    shrink_dataset_by_label_filtering(
        labels_to_use=("HealthyControl", "BundleBranchBlock", "Myocarditis"),
        dirname="ptb-matrices",
        name="all_samples_3_diseases"
    )
    # shrink_dataset_by_number_and_filter(
    #     train_size=3000,
    #     test_size=600,
    #     labels_to_use=("HealthyControl", "BundleBranchBlock", "Myocarditis"),
    #     dirname="ptb-concatted-leads",
    #     name="3k_per_disease_3_diseases"
    # )
