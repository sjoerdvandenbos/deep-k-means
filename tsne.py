from argparse import ArgumentParser
from pathlib import Path
import re

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from utils import get_color_map, get_dataset_dir, read_list


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_directory", type=str, required=True)
    parser.add_argument("-f", "--label_file", type=str, default="compacted_target.npy")
    parser.add_argument("-a", "--plotting_alpha", type=float, default=0.5)
    parser.add_argument("-p", "--perplexity", type=float, default=30.0)
    parser.add_argument("-l", "--learning_rate", type=float, default=200.0)
    parser.add_argument("-n", "--n_iter", type=int, default=1000)
    parser.add_argument("-e", "--early_exaggeration", type=float, default=12)
    parser.add_argument("-m", "--metric", type=str, default="euclidean")
    args = parser.parse_args()
    str_repr = f"perplexity{args.perplexity}_lr{args.learning_rate}_n{args.n_iter}_early{args.early_exaggeration}," \
               f" metric{args.metric}_label{Path(args.label_file).name}"
    return args, str_repr


def get_tsne_mapped_data(metrics_dir, perplexity, learning_rate, n_iter, early_exaggeration, metric):
    """
    For different metric values see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html.
    """
    embeds = np.load(metrics_dir / "test_embeds0.npy")
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter,
                early_exaggeration=early_exaggeration, init="pca", metric=metric, square_distances=True)
    return tsne.fit_transform(embeds)


def plot(mapped_embeds, ground_truths, directory, args_str, alpha):
    fig, ax = plt.subplots(1, 1)
    inverse_label_mapping = dict(enumerate(np.unique(ground_truths)))
    # maps label to int
    label_mapping = {v: k for k, v in inverse_label_mapping.items()}
    # maps int to color code
    color_map = get_color_map(len(label_mapping))
    color_labels = np.array([color_map[label_mapping[i]] for i in ground_truths])
    for i in np.arange(mapped_embeds.shape[0]):
        ax.plot(mapped_embeds[i, 0], mapped_embeds[i, 1], label=ground_truths[i], marker=".", color=color_labels[i],
                alpha=alpha)
    _legend_without_duplicate_labels(ax, bbox_to_anchor=(0.75, 1), loc="upper left", fancybox=True)

    root = Path(__file__).parent
    plot_number = _get_next_plot_number(directory)
    fig.savefig(directory / f"tsne_plot{plot_number}_{args_str}.jpg")
    fig.clear()
    print(f"Plot {plot_number} saved.")


def _get_next_plot_number(directory):
    regex = r"^tsne_plot([0-9]+)_.*\.jpg$"
    state_machine = re.compile(regex)
    existing_plot_names = [p.name for p in directory.glob("tsne_plot*")]
    if len(existing_plot_names) > 0:
        existing_numbers = [int(state_machine.match(n).group(1)) for n in existing_plot_names]
        return max(existing_numbers) + 1
    else:
        return 0


def _legend_without_duplicate_labels(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    for h, l in unique:
        h.set_alpha(1)
    ax.legend(*zip(*unique), **kwargs)


def _dir_contains(directory, file_name):
    return (directory / file_name).exists()


if __name__ == "__main__":
    args, args_str = parse_args()
    dir = Path(__file__).parent / "metrics" / args.data_directory
    data = get_tsne_mapped_data(dir, args.perplexity, args.learning_rate, args.n_iter,
                                args.early_exaggeration, args.metric)
    dataset_dir = get_dataset_dir(dir)
    test_indices = read_list(dir / "validation.csv")
    all_targets = np.load(dataset_dir / "compacted_target.npy")
    targets = all_targets[test_indices]
    plot(data, targets, dir, args_str, args.plotting_alpha)
