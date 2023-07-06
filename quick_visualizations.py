from pathlib import Path
import argparse
import re

from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from utils import get_color_map, read_list


def visualize(embeds, name, directory, run):
    targets = read_list(get_patient_numbers(directory))
    unique = np.unique(targets)
    n = 10
    first_n = unique[:n]
    inverse_class_mapping = dict(enumerate(unique))
    class_mapping = {v: k for k, v in inverse_class_mapping.items()}
    darkened_color_map = get_color_map(n, is_darker=True)
    others = {num: "black" for num in np.arange(n, len(unique))}
    darkened_color_map.update(others)
    embeds_colors = [darkened_color_map[class_mapping[t]] for t in targets]
    fig, ax = plt.subplots()
    for i in range(embeds.shape[0]):
        if targets[i] in first_n:
            ax.plot(embeds[i, 0], embeds[i, 1], ".", color=embeds_colors[i],
                    label=f"{targets[i]}", alpha=0.3)
        else:
            ax.plot(embeds[i, 0], embeds[i, 1], ".", color="#000000", alpha=0.001)
    # Shrink current axis's height by 10% on the bottom
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    # _legend_without_duplicate_labels(ax, loc="upper center", bbox_to_anchor=(0.5, -0.05), fancybox=True)
    fig.savefig(directory / f"torch_centers_run{run}_{name}.jpg")
    fig.clear()


def _legend_without_duplicate_labels(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    for h, l in unique:
        h.set_alpha(1)
    ax.legend(*zip(*unique), **kwargs)


def get_patient_numbers(dir):
    log = dir / "log.txt"
    regex = r"^.*dataset_path=(.*),.*$"
    state_machine = re.compile(regex)
    for line in log.open("r"):
        if state_machine.match(line):
            data_dir = state_machine.match(line).group(1)
            return Path(data_dir) / "patient_numbers.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory_name", type=str, required=True)
    args = parser.parse_args()
    directory = Path(__file__).parent / "metrics" / args.directory_name
    embeds = np.load(directory / "test_embeds0.npy")
    tsne_embeds = TSNE(n_components=2, init="pca").fit_transform(embeds)
    visualize(tsne_embeds, "patient_numbers", directory, 0)
