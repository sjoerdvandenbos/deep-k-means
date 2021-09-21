from pathlib import Path
from re import compile
from inspect import ismethod, getmembers

import matplotlib.pyplot as plt
import numpy as np


ACC_REGEX = compile(r"^(?:Test|Train) ACC: (?P<acc>[01]\.[0-9]+$)")
ARI_REGEX = compile(r"ARI: (?P<ari>.*$)")
NMI_REGEX = compile(r"NMI: (?P<nmi>.*$)")
LOSS_REGEX = compile(r"^(?:Train|Test) loss: (?P<loss>.*$)")
AE_LOSS_REGEX = compile(r"[Aa]uto encoder loss: (?P<ae_loss>.*$)")
KMEANS_LOSS_REGEX = compile(r"kmeans loss: (?P<kmeans_loss>.*$)")

FINETUNING_REGEX = compile(r"^Training step: epoch [0-9]+$")
PRETRAINING_REGEX = compile(r"^Pretraining step: epoch [0-9]+$")
PRETRAIN_ACCIDENT_REGEX = compile(r"^Auto encoder loss: .*$")


def read_log(filename):
    log = open(Path.cwd() / filename, "r")
    pretrain_metrics = fill_list(Metrics, 10)
    pretest_metrics = fill_list(Metrics, 10)
    finetrain_metrics = fill_list(Metrics, 10)
    finetest_metrics = fill_list(Metrics, 10)
    pretraining = False
    run = -1
    for line in log:
        if "Run" in line:
            run += 1

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
    if "Train" in line or PRETRAIN_ACCIDENT_REGEX.match(line):
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


def plot(metrics, phase):
    for name in metrics.get_field_names():
        metric_series = getattr(metrics, name)
        plt.plot(metric_series)
        plt.title(f"{phase} {name}")
        plt.savefig(Path.cwd() / "metrics" / f"{phase}_{name}.jpg")
        plt.close()


def average_runs(metrics_list, exclude_fields=[]):
    """
    @param metrics_list: A python list containing n instances of the Metrics class.
    @returns: A single instance of the Metrics class with metrics which are averaged
    over all items n items.
    """
    averaged = Metrics()
    fields = [field for field in averaged.get_field_names() if field not in exclude_fields]
    runs = np.arange(0, 10)
    samples_per_series = 50
    for field in fields:
        new = np.zeros((runs.shape[0], samples_per_series))
        for run in runs:
            new[run, :] = getattr(metrics_list[run], field)
        setattr(averaged, field, np.mean(new, axis=0))
    return averaged


def fill_list(filling, length):
    """ Instanciates a list with specified filling and length. """
    return [filling() for _ in range(length)]


def map_list_to_numpy(metrics_list):
    """
    Calls convert_to_numpy on all metrics objects in this list in place.
    @returns: a pointer to the same list which was used as input.
    """
    for metrics in metrics_list:
        metrics.convert_to_numpy()
    return metrics_list


class Metrics:
    def __init__(self):
        self.losses = []
        self.ae_losses = []
        self.kmeans_losses = []
        self.accuracies = []
        self.aris = []
        self.nmis = []

    def convert_to_numpy(self):
        self.losses = np.asarray(self.losses, dtype=float)
        self.ae_losses = np.asarray(self.ae_losses, dtype=float)
        self.kmeans_losses = np.asarray(self.kmeans_losses, dtype=float)
        self.accuracies = np.asarray(self.accuracies, dtype=float)
        self.aris = np.asarray(self.aris, dtype=float)
        self.nmis = np.asarray(self.nmis, dtype=float)

    def get_field_names(self):
        return [k for k, v in getmembers(Metrics())
                if not k.startswith("_") and not ismethod(v)]


if __name__ == "__main__":
    path = Path.cwd() / "mnist_reprod_log.txt"
    print(f"Reading from {path}")
    pretrain, pretest, finetrain, finetest = read_log(path)

    pretrain_avg = average_runs(pretrain, exclude_fields=["losses", "kmeans_losses"])
    pretest_avg = average_runs(pretest, exclude_fields=["losses", "kmeans_losses", "ae_losses"])
    finetrain_avg = average_runs(finetrain)
    finetest_avg = average_runs(finetest)

    plot(pretrain_avg, "pretrain")
    plot(pretest_avg, "pretest")
    plot(finetrain_avg, "finetrain")
    plot(finetest_avg, "finetest")
