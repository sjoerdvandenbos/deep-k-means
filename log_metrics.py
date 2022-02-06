from pathlib import Path
from re import compile
from inspect import ismethod, getmembers

import matplotlib.pyplot as plt
import numpy as np


ACC_REGEX = compile(r"^(?:Test|Train) ACC: (?P<acc>[01]\.[0-9]+$)")
ARI_REGEX = compile(r"ARI: (?P<ari>.*$)")
NMI_REGEX = compile(r"NMI: (?P<nmi>.*$)")
LOSS_REGEX = compile(r"^(?:Train|Test) loss: (?P<loss>.*$)")
AE_LOSS_REGEX = compile(r"^(?:Train|Test) auto encoder loss: (?P<ae_loss>.*$)")
KMEANS_LOSS_REGEX = compile(r"kmeans loss: (?P<kmeans_loss>.*$)")

FINETUNING_REGEX = compile(r"^Training step: epoch [0-9]+$")
PRETRAINING_REGEX = compile(r"^Pretraining step: epoch [0-9]+$")


def write_visuals_and_summary(folder):
    logfile = folder / "log.txt"
    print(f"Reading from {logfile}")
    pretrain, pretest, finetrain, finetest = read_log(logfile)
    n_finetune_epochs = len(finetrain[0])

    pretrain_avg, pretrain_stddev = average_runs(pretrain, exclude_fields=("losses", "kmeans_losses"))
    pretest_avg, pretest_stddev = average_runs(pretest, exclude_fields=("losses", "kmeans_losses"))
    plot(pretrain, pretrain_avg, pretrain_stddev, "pretrain", folder)
    plot(pretest, pretest_avg, pretest_stddev, "pretest", folder)

    if n_finetune_epochs > 0:
        finetrain_avg, finetrain_stddev = average_runs(finetrain)
        finetest_avg, finetest_stddev = average_runs(finetest)
        plot(finetrain, finetrain_avg, finetrain_stddev, "finetrain", folder)
        plot(finetest, finetest_avg, finetest_stddev, "finetest", folder)

    summary_lines = []
    summary_lines.extend(summarize_results(
        pretrain_avg, pretrain_stddev, "pretrain", exclude_fields=("losses", "kmeans_losses")))
    summary_lines.extend(summarize_results(
        pretest_avg, pretest_stddev, "pretest", exclude_fields=("losses", "kmeans_losses")))
    if n_finetune_epochs > 0:
        summary_lines.extend(summarize_results(finetrain_avg, finetrain_stddev, "finetrain"))
        summary_lines.extend(summarize_results(finetest_avg, finetest_stddev, "finetest"))
    summary_file = folder / "summary.txt"
    with summary_file.open("w+") as file:
        file.writelines(summary_lines)


def read_log(filename):
    log = open(Path.cwd() / filename, "r")
    pretrain_metrics = []
    pretest_metrics = []
    finetrain_metrics = []
    finetest_metrics = []
    pretraining = False
    run = -1
    for line in log:
        # Don't read before the training started
        if run == -1 and "Run" not in line:
            continue
        if "Run" in line:
            run += 1
            pretrain_metrics.append(Metrics())
            pretest_metrics.append(Metrics())
            finetrain_metrics.append(Metrics())
            finetest_metrics.append(Metrics())

        if PRETRAINING_REGEX.match(line):
            pretraining = True
        elif FINETUNING_REGEX.match(line):
            pretraining = False

        if pretraining:
            read_train_or_test_line(line, pretrain_metrics[run], pretest_metrics[run])
        else:
            read_train_or_test_line(line, finetrain_metrics[run], finetest_metrics[run])
    return (
        map_list_to_numpy(pretrain_metrics),
        map_list_to_numpy(pretest_metrics),
        map_list_to_numpy(finetrain_metrics),
        map_list_to_numpy(finetest_metrics),
    )


def read_train_or_test_line(line, train_metrics, test_metrics):
    if "Train" in line:
        read_line(line, train_metrics)
    elif "Test" in line:
        read_line(line, test_metrics)


def read_line(line, metrics):
    if ACC_REGEX.match(line):
        metrics.accuracies.append(ACC_REGEX.match(line).group("acc"))
    elif ARI_REGEX.search(line):
        metrics.aris.append(ARI_REGEX.search(line).group("ari"))
    elif NMI_REGEX.search(line):
        metrics.nmis.append(NMI_REGEX.search(line).group("nmi"))
    elif LOSS_REGEX.search(line):
        metrics.losses.append(LOSS_REGEX.search(line).group("loss"))
    elif AE_LOSS_REGEX.search(line):
        metrics.ae_losses.append(AE_LOSS_REGEX.search(line).group("ae_loss"))
    elif KMEANS_LOSS_REGEX.search(line):
        metrics.kmeans_losses.append(KMEANS_LOSS_REGEX.search(line).group("kmeans_loss"))


def plot(metrics, means, stddev, phase, folder):
    for name in means.get_field_names():
        n_examples = min(3, len(metrics))
        rand_indices = np.random.choice(range(len(metrics)), n_examples)
        for i in rand_indices:
            example_sample = getattr(metrics[i], name)
            plt.plot(example_sample, color="blue")
        means_series = getattr(means, name)
        deviation = getattr(stddev, name)
        x = np.arange(0, len(means_series))
        plt.fill_between(x, means_series + 2*deviation, means_series - 2*deviation, alpha=0.3)
        plt.xlabel("epochs")
        plt.title(f"{phase} {name}")
        plt.savefig(folder / f"{phase}_{name}.jpg")
        plt.close()


def average_runs(metrics_list, exclude_fields=()):
    """
    @param metrics_list: A python list containing n instances of the Metrics class.
    @returns: A single instance of the Metrics class with metrics which are averaged
    over all items n items.
    """
    averaged = Metrics()
    stddev = Metrics()
    fields = [field for field in averaged.get_field_names() if field not in exclude_fields]
    runs = np.arange(0, len(metrics_list))
    samples_per_series = len(getattr(metrics_list[0], "accuracies"))
    for field in fields:
        new = np.zeros((runs.shape[0], samples_per_series))
        for run in runs:
            new[run, :] = getattr(metrics_list[run], field)
        setattr(averaged, field, np.mean(new, axis=0))
        variation = np.mean(new**2, axis=0) - np.mean(new, axis=0)**2
        setattr(stddev, field, np.sqrt(variation))
    return averaged.convert_to_numpy(), stddev.convert_to_numpy()


def summarize_results(avg_metrics, std_metrics, phase, exclude_fields=()):
    summary_lines = [
        f"{phase}\n",
        "---------------------------\n"
    ]
    for field in avg_metrics.get_field_names():
        if field not in exclude_fields:
            avg = getattr(avg_metrics, field)[-1]
            std = getattr(std_metrics, field)[-1]
            result = f"{field}: {round(avg, 4)} +- {round(std, 4)}\n"
            summary_lines.append(result)
    summary_lines.append("\n")
    return summary_lines


def map_list_to_numpy(metrics_list):
    return [m.convert_to_numpy() for m in metrics_list]


def fix_ae_loss(metrics_list):
    for m in metrics_list:
        m.ae_losses = m.losses - m.kmeans_losses


class Metrics:
    def __init__(self):
        self.losses = []
        self.ae_losses = []
        self.kmeans_losses = []
        self.accuracies = []
        self.aris = []
        self.nmis = []

    def convert_to_numpy(self):
        result = Metrics()
        result.losses = np.array(self.losses, dtype=float)
        result.ae_losses = np.array(self.ae_losses, dtype=float)
        result.kmeans_losses = np.array(self.kmeans_losses, dtype=float)
        result.accuracies = np.array(self.accuracies, dtype=float)
        result.aris = np.array(self.aris, dtype=float)
        result.nmis = np.array(self.nmis, dtype=float)
        return result

    def deep_copy(self):
        new = Metrics()
        for field in self.get_field_names():
            copy = np.array(getattr(self, field), copy=True)
            setattr(new, field, copy)
        return new

    def get_field_names(self):
        return [k for k, v in getmembers(self)
                if not k.startswith("_") and not ismethod(v)]

    def __len__(self):
        return len(self.accuracies)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parser for generation of visuals and summary")
    parser.add_argument("-d", "--directory", type=str, required=True, dest="directory")
    args = parser.parse_args()
    path = Path.cwd() / "metrics" / args.directory
    write_visuals_and_summary(path)
