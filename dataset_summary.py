from pathlib import Path
from json import dumps, JSONEncoder

import numpy as np
import matplotlib.pyplot as plt

from utils import read_list


def summarize(path) -> None:
    train_indices = read_list(path / "train")
    test_indices = read_list(path / "validation")
    data = np.load(path / "compacted_data.npy")
    train_data = data[train_indices]
    test_data = data[test_indices]
    train_metrics = _get_metrics(train_data)
    test_metrics = _get_metrics(test_data)
    train_distribution = _get_label_distribution(path, train_indices)
    test_distribution = _get_label_distribution(path, test_indices)
    _plot(path / "train_distribution.jpg", train_distribution)
    _plot(path / "test_distribution.jpg", test_distribution)
    with path.joinpath("dataset_exploration.txt").open("w+") as f:
        f.writelines(_get_summary(train_distribution, test_distribution, train_metrics, test_metrics))


def _get_summary(train_distribution, test_distribution, train_metrics, test_metrics) -> tuple:
    train_summary = dumps(train_distribution, indent=4, cls=Int32Encoder)
    test_summary = dumps(test_distribution, indent=4, cls=Int32Encoder)
    total = np.fromiter((*train_distribution.values(), *test_distribution.values()), dtype=int).sum()
    separator = "\n---------------------------\n"
    return f"total: {total}", separator, train_metrics, train_summary, separator, test_metrics, test_summary, separator


def _get_metrics(data):
    contains_nan = np.isnan(data.sum())
    return f"min: {np.min(data)}, max: {np.max(data)}, avg: {np.average(data)}, contains NaNs: {contains_nan}," \
           f" shape: {data.shape}\n"


def _plot(img_path, distribution) -> None:
    values = np.fromiter(distribution.values(), dtype=int)
    _, _, autotexts = plt.pie(values, labels=distribution.keys(), autopct="")
    # Add label frequencies to plot
    for i, a in enumerate(autotexts):
        a.set_text(f"{values[i]}")
    plt.savefig(img_path)
    plt.close()


def _get_label_distribution(path, indices) -> dict:
    labels = np.load(path / "compacted_target.npy")
    keys = np.unique(labels)
    values = np.zeros_like(keys, dtype=int)
    label_distribution = dict(zip(keys, values))
    for index in indices:
        label = labels[index]
        label_distribution[label] += 1
    return label_distribution


class Int32Encoder(JSONEncoder):
    """ Default JSON encoder can not deal with np.int32. """
    def default(self, obj):
        if isinstance(obj, np.int32):
            return str(obj)
        else:
            super().default(obj)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", type=str, help="path from the split directory to dataset",
                        required=True)
    args = parser.parse_args()
    data_path = Path(__file__).parent / "split" / args.dataset_path
    summarize(data_path)
