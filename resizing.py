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
    new_train_indices = get_resized_indices(train_size, old_train_indices, target)
    new_test_indices = get_resized_indices(test_size, old_test_indices, target)
    write_smaller_npy(split_path / "ptb-images-2-cropped", new_train_indices, new_test_indices, name)
    write_smaller_npy(split_path / "ptb-matrices", new_train_indices, new_test_indices, name)


def get_resized_indices(new_size, old_indices, target):
    path_iterator = (Path.cwd() / "split" / "ptb-matrices" / "train").glob("*")
    disease_names = [e.name for e in path_iterator]
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


def write_smaller_npy(path, train_indices, test_indices, name):
    data, target = load_and_shrink_npy(path, train_indices, test_indices)
    destination = path / name
    destination.mkdir(parents=False, exist_ok=True)
    save_npy(destination, data, target)
    # Write train indices
    pd.DataFrame(np.arange(0, len(train_indices))).to_csv(destination / "validation", index=False, header=False)
    # Write test indices
    first_test_index = len(train_indices)
    last_test_index = len(train_indices) + len(test_indices)
    pd.DataFrame(np.arange(first_test_index, last_test_index)).to_csv(destination / "test", index=False, header=False)


def load_and_shrink_npy(path, train_indices, test_indices):
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
    np.save(path / f"compacted_data", data)
    np.save(path / f"compacted_target", target)


if __name__ == "__main__":
    shrink_imgs_and_matrices(train_size=10000, test_size=2000, name="10k_per_disease")
