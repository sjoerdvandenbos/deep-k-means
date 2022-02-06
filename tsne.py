from argparse import ArgumentParser
from pathlib import Path
import re

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from utils import load_dataset, get_color_map


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_directory", type=str, required=True)
    parser.add_argument("-p", "--perplexity", type=float, default=30.0)
    parser.add_argument("-l", "--learning_rate", type=float, default=200.0)
    parser.add_argument("-n", "--n_iter", type=int, default=1000)
    parser.add_argument("-e", "--early_exaggeration", type=float, default=12)
    parser.add_argument("-m", "--metric", type=str, default="euclidian")
    args = parser.parse_args()
    str_repr = f"perplexity{args.perplexity}_lr{args.learning_rate}_n{args.n_iter}_early{args.early_exaggeration}," \
               f" metric{args.metric}"
    return args, str_repr


def get_tsne_mapped_data(directory, perplexity, learning_rate, n_iter, early_exaggeration, metric):
    """
    For different metric values see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html.
    """
    root = Path(__file__).parent
    embeds = np.load(root / "metrics" / directory / "test_embeds.npy")
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter,
                early_exaggeration=early_exaggeration, init="pca", metric=metric, square_distances=True)
    return tsne.fit_transform(embeds)


def get_ground_truths():
    root = Path(__file__).parent
    _, target, _, test_indices = load_dataset(root / "split" / "ptb-12lead-matrices-gaussian-normalized-100hz"
                                              / "all_samples_2_diseases")
    return target[test_indices]


def plot(mapped_embeds, ground_truths, directory, args_str):
    inverse_label_mapping = dict(enumerate(np.unique(ground_truths)))
    # maps label to int
    label_mapping = {v: k for k, v in inverse_label_mapping.items()}
    # maps int to color code
    color_map = get_color_map(len(label_mapping))
    color_labels = np.array([color_map[label_mapping[i]] for i in ground_truths])
    for i in np.arange(mapped_embeds.shape[0]):
        plt.plot(mapped_embeds[i, 0], mapped_embeds[i, 1], marker=".", color=color_labels[i], alpha=0.1)

    root = Path(__file__).parent
    target_dir = root / "metrics" / directory
    plot_number = get_next_plot_number(target_dir)
    plt.savefig(target_dir / f"tsne_plot{plot_number}_{args_str}.jpg")
    print(f"Plot {plot_number} saved.")


def get_next_plot_number(directory):
    regex = r"^tsne_plot([0-9]+)_.*\.jpg$"
    state_machine = re.compile(regex)
    existing_plot_names = [p.name for p in directory.glob("tsne_plot*")]
    if len(existing_plot_names) > 0:
        existing_numbers = [int(state_machine.match(n).group(1)) for n in existing_plot_names]
        return max(existing_numbers) + 1
    else:
        return 0


if __name__ == "__main__":
    args, args_str = parse_args()
    data = get_tsne_mapped_data(args.data_directory, args.perplexity, args.learning_rate, args.n_iter,
                                args.early_exaggeration, args.metric)
    targets = get_ground_truths()
    plot(data, targets, args.data_directory, args_str)
