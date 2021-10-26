#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

import math
from pathlib import Path
from datetime import datetime
import numpy as np
import argparse

import torch.cuda
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torch
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from matplotlib.colors import TABLEAU_COLORS

from utils import cluster_acc, map_clusterlabels_to_groundtruth
from ptb_img_utils import reconstruct_image, DISEASE_MAPPING
from Modules import DeepKMeans

INVERSE_DISEASE_MAPPING = {v: k for k, v in DISEASE_MAPPING.items()}


def print_cluster_metrics(ground_truths, cluster_labels, phase, write_files, logfile):
    """
    @param phase: The phase of the experiment, either train or test.
    """
    predictions = map_clusterlabels_to_groundtruth(ground_truths, cluster_labels)
    acc = cluster_acc(ground_truths, predictions)
    log(f"{phase} ACC: {acc}", write_files, logfile)
    ari = adjusted_rand_score(ground_truths, predictions)
    log(f"{phase} ARI: {ari}", write_files, logfile)
    nmi = normalized_mutual_info_score(ground_truths, predictions)
    log(f"{phase} NMI: {nmi}", write_files, logfile)


def infer_cluster_label(distances):
    return torch.argmin(distances, dim=1)


def log(content, to_disc, path):
    print(content)
    if to_disc:
        with path.open("a+") as f:
            f.write(f"{content}\n")


def map_range_to_01(x):
    """ Returns imput tensor value range to [0, 1] """
    return torch.sigmoid(x)


def f1_loss(iput, reconstruction, epsilon=1e-8):
    # Iput range: [0, 1]
    # Reconstruction range: unknown
    reconstruction_binary = map_range_to_01(reconstruction)
    intersection = torch.sum(iput * reconstruction_binary, dim=0)
    denominator = torch.sum(iput + reconstruction_binary, dim=0)
    f1 = (2. * intersection + epsilon) / (denominator + epsilon)
    return torch.mean(1. - f1)


def jaccard_loss(iput, reconstruction, epsilon=1e-8):
    # Iput range: [0, 1]
    # Reconstruction range: unknown
    reconstruction_binary = map_range_to_01(reconstruction)
    intersection = iput * reconstruction_binary
    union = (iput + reconstruction_binary) - intersection
    jac = torch.sum((intersection + epsilon) / (union + epsilon), dim=0)
    return torch.mean(1. - jac)


parser = argparse.ArgumentParser(description="Deep k-means algorithm")
parser.add_argument("-d", "--dataset", type=str.upper,
                    help="Dataset on which DKM will be run (one of USPS, MNIST, 20NEWS, RCV1)", required=True)
parser.add_argument("-s", "--seeded", help="Use a fixed seed, different for each run", action='store_true')
parser.add_argument("-c", "--cpu", help="Force the program to run on CPU", action='store_true')
parser.add_argument("-l", "--lambda", type=float, default=1.0, dest="lambda_",
                    help="Value of the hyperparameter weighing the clustering loss against the reconstruction loss")
parser.add_argument("-e", "--p-epochs", type=int, default=50, help="Number of pretraining epochs")
parser.add_argument("-f", "--f-epochs", type=int, default=5, help="Number of fine-tuning epochs per alpha value")
parser.add_argument("-b", "--batch_size", type=int, default=256, help="Size of the minibatches used by the optimizer")
parser.add_argument("-n", "--number_runs", type=int, default=1, help="number of repetitions of the entire experiment")
parser.add_argument("-a", "--autoencoder", type=str, default="linear", help="type of autoencoder to use")
parser.add_argument("-w", "--write_files", default=False, action="store_true", help="if enabled, "
                                                                                               "will write files to "
                                                                                               "disk")
parser.add_argument("-o", "--loss", type=str, default="MSE", help="type of loss function")
args = parser.parse_args()

# Dataset setting from arguments
if args.dataset == "MNIST":
    import mnist_specs as specs
elif args.dataset == "PTB":
    import ptb_img_specs as specs
elif args.dataset == "PTBMAT":
    import ptb_matrix_specs as specs
else:
    parser.error("Unknown dataset!")
    exit()

# Autoencoder setting from arguments
if args.autoencoder == "convolutional":
    from Modules import ConvoAutoencoder
    autoencoder = ConvoAutoencoder(specs.img_height, specs.img_width, specs.n_clusters)
elif args.autoencoder == "OLM":
    from Modules import OLMAutoencoder
    autoencoder = OLMAutoencoder(specs.dimensions, specs.img_height, specs.img_width)
else:
    from Modules import FCAutoencoder
    autoencoder = FCAutoencoder(specs.dimensions, specs.img_height, specs.img_width)

# AE loss setting from arguments
if args.loss == "f1":
    autoencoder_loss = f1_loss
elif args.loss == "jaccard":
    autoencoder_loss = jaccard_loss
else:
    from ptb_img_utils import autoencoder_loss

# Parameter setting from arguments
n_pretrain_epochs = args.p_epochs
n_finetuning_epochs = args.f_epochs
lambda_ = args.lambda_                                      # Value of hyperparam lambda balancing ae and kmeans losses
batch_size = args.batch_size                                # Size of the mini-batches used in the stochastic optimizer
test_size = batch_size
trainset, testset = specs.trainset, specs.testset
n_batches = int(math.ceil(len(trainset) / batch_size))    # Number of mini-batches
n_test_batches = int(math.ceil(len(testset) / test_size))
seeded = args.seeded                                        # Specify if runs are seeded
n_runs = args.number_runs
write_files = args.write_files

now = datetime.now()
time_format = "%Y_%m_%dT%H_%M"
experiment_id = f"{args.dataset}_e_{n_pretrain_epochs}_f_{n_finetuning_epochs}_bs_{batch_size}_" \
                f"{now.strftime(time_format)}"
directory = Path.cwd() / "metrics" / experiment_id
if write_files:
    directory.mkdir()
logfile = directory / "log.txt"

log(f"Hyperparameters: lambda={lambda_}, pretrain_epochs={n_pretrain_epochs}, finetune_epochs={n_finetuning_epochs}, "
    f"batch_size={batch_size}, autoencoder={args.autoencoder}, loss={args.loss}, lambda={args.lambda_}", write_files,
    logfile)

# Hardware specifications
device = "cuda" if torch.cuda.is_available() else "cpu"
if args.cpu:
    device = "cpu"
num_workers = 0

# Definition of the randomly-drawn (0-10000) seeds to be used for each run
seeds = [8905, 9129, 291, 4012, 1256, 6819, 4678, 6971, 1362, 575]

for run in range(n_runs):
    log(f"Run {run}", write_files, logfile)
    if seeded:
        torch.manual_seed(seeds[run])
        np.random.seed(seeds[run])
    autoencoder = autoencoder.to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), 0.001)
    kmeans_model = None

    for epoch in range(n_pretrain_epochs):
        log(f"Pretraining step: epoch {epoch}", write_files, logfile)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        train_embeddings = torch.zeros((len(trainset), specs.embedding_size), dtype=torch.float)
        ae_losses = []
        autoencoder.train()
        for train_indices, train_batch, _ in train_loader:
            train_batch = train_batch.to(device)

            # Train one mini-batch
            optimizer.zero_grad()
            train_embedding_, reconstruction = autoencoder.forward(train_batch)
            train_ae_loss_ = autoencoder_loss(train_batch, reconstruction)
            train_ae_loss_.backward()
            optimizer.step()

            # Save metrics to cpu memory
            ae_losses.append(train_ae_loss_.item())
            train_embeddings[train_indices, :] = train_embedding_.detach().cpu()

        del train_embedding_, reconstruction, train_ae_loss_, train_batch

        test_loader = DataLoader(testset, batch_size=test_size, shuffle=True)
        test_embeddings = torch.zeros((len(testset), specs.embedding_size), dtype=torch.float)
        test_ae_losses = []
        autoencoder.eval()
        with torch.no_grad():
            for test_indices, test_batch, _ in test_loader:
                test_batch = test_batch.to(device)

                # Eval one mini-batch
                test_embedding_, test_reconstruction = autoencoder.forward(test_batch)
                test_ae_loss = autoencoder_loss(test_batch, test_reconstruction)

                # Save metrics to cpu memory
                test_ae_losses.append(test_ae_loss.item())
                test_embeddings[test_indices, :] = test_embedding_.detach().cpu()

        kmeans_model = KMeans(n_clusters=specs.n_clusters, init="k-means++").fit(train_embeddings)
        train_cluster_labels = torch.from_numpy(kmeans_model.labels_).long()
        test_cluster_labels = torch.from_numpy(kmeans_model.predict(test_embeddings))

        log(f"Train auto encoder loss: {sum(ae_losses) / n_batches}", write_files, logfile)
        print_cluster_metrics(trainset.target, train_cluster_labels, "Train", write_files, logfile)
        log(f"Test auto encoder loss: {sum(test_ae_losses) / n_test_batches}", write_files, logfile)
        print_cluster_metrics(testset.target, test_cluster_labels, "Test", write_files, logfile)
        log("", write_files, logfile)

    # Visualize auto encoder input and respective output
    if write_files:
        img_shape = [specs.img_height, specs.img_width]
        now = datetime.now()
        time_format = "%Y_%m_%dT%H_%M"
        index = np.random.choice(len(testset), size=1)
        rand_input = testset[index][1].to(device)
        with torch.no_grad():
            reconstruction = autoencoder.forward(rand_input)[1].cpu().detach().numpy()
        rand_input = rand_input.cpu().detach().numpy()
        input_img = reconstruct_image(rand_input, img_shape).convert("L")
        output_img = reconstruct_image(reconstruction, img_shape, transform=True).convert("L")
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(input_img)
        ax1.set_title("input")
        ax2.imshow(output_img)
        ax2.set_title("reconstruction")
        fig.savefig(directory / f"input_e{n_pretrain_epochs}_bs{batch_size}_{now.strftime(time_format)}.jpg")

    # Visualize embedding space using t-SNE
    if write_files:
        centers = torch.from_numpy(kmeans_model.cluster_centers_)
        test_embeds_and_centers = torch.cat((test_embeddings, centers))
        fitted = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(test_embeds_and_centers)
        mapped_embeds, mapped_centers = fitted[:len(testset), :], fitted[len(testset):, :]
        colors = [*TABLEAU_COLORS.values()]
        color_map = {i: colors[i] for i in range(specs.embedding_size)}
        color_labels = [color_map[t.item()] for t in testset.target]
        fig2, ax3 = plt.subplots()
        for i in range(len(testset)):
            ax3.plot(mapped_embeds[i, 0], mapped_embeds[i, 1], ".", color=color_labels[i])
        for j in range(len(mapped_centers)):
            if args.dataset == "MNIST":
                ax3.plot(mapped_centers[j, 0], mapped_centers[j, 1], "o", color=color_map[j], label=f"{j}")
            else:
                ax3.plot(
                    mapped_centers[j, 0], mapped_centers[j, 1], "o", color=color_map[j]
                    , label=f"{INVERSE_DISEASE_MAPPING[j]}")
        ax3.legend(loc="upper right")
        fig2.savefig(directory / f"torch_centers{now.strftime(time_format)}.jpg")

    # Train the full DKM model
    cluster_centers = torch.from_numpy(kmeans_model.cluster_centers_).to(device)
    deepkmeans = DeepKMeans(autoencoder, cluster_centers)
    optimizer = torch.optim.Adam(deepkmeans.parameters(), 0.001)
    for epoch in range(n_finetuning_epochs):
        train_distances = torch.zeros((len(trainset), specs.embedding_size))
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        train_kmeans_losses, train_ae_losses = [], []

        deepkmeans.train()
        log(f"Training step: epoch {epoch}", write_files, logfile)
        for train_indices, train_batch, _ in train_loader:
            train_batch = train_batch.to(device)

            # Train one mini-batch
            optimizer.zero_grad()
            train_kmeans_distance_, stack_dist, train_reconstruction = deepkmeans.forward(train_batch)
            # print(f"train_kmeans_distance shape: {train_kmeans_distance_.shape}")
            # print(f"train_kmeans_distance: {train_kmeans_distance_}")
            # print(f"sum kmeans: {train_kmeans_distance_.sum(dim=1)}")
            # exit()
            train_ae_loss_ = autoencoder_loss(train_batch, train_reconstruction)
            train_kmeans_loss_ = train_kmeans_distance_.sum(dim=1).mean()
            total_train_loss_ = train_ae_loss_ + lambda_ * train_kmeans_loss_
            total_train_loss_.backward()
            optimizer.step()

            # Save metrics do cpu memory
            train_kmeans_losses.append(train_kmeans_loss_.item())
            train_ae_losses.append(train_ae_loss_.item())
            train_distances[train_indices, :] = stack_dist.detach().cpu().float()

        del train_batch, train_kmeans_distance_, stack_dist, train_reconstruction, train_ae_loss_, \
            train_kmeans_loss_, total_train_loss_

        train_cluster_labels = infer_cluster_label(train_distances)
        print_cluster_metrics(trainset.target, train_cluster_labels, "Train", write_files, logfile)
        train_ae, train_kmeans = sum(train_ae_losses) / n_batches, sum(train_kmeans_losses) / n_batches
        log(f"Train loss: {train_ae + lambda_ * train_kmeans}", write_files, logfile)
        log(f"Train auto encoder loss: {train_ae}", write_files, logfile)
        log(f"Train kmeans loss: {train_kmeans}", write_files, logfile)

        test_distances = torch.zeros((len(testset), specs.embedding_size))
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
        test_kmeans_losses, test_ae_losses = [], []
        deepkmeans.eval()
        with torch.no_grad():
            for test_indices, test_batch, _ in test_loader:
                test_batch = test_batch.to(device)

                # Eval one mini-batch
                test_kmeans_distance, test_stack_dist, test_reconstruction = deepkmeans.forward(test_batch)
                test_ae_loss = autoencoder_loss(test_batch, test_reconstruction).item()
                test_kmeans_loss = test_kmeans_distance.sum(dim=1).mean().item()
                total_test_loss = test_ae_loss + lambda_ * test_kmeans_loss

                # Save metrics to cpu memory
                test_ae_losses.append(test_ae_loss)
                test_kmeans_losses.append(test_kmeans_loss)
                test_distances[test_indices, :] = test_stack_dist.detach().cpu().float()

            del test_batch, test_kmeans_distance, test_stack_dist, test_reconstruction, test_ae_loss, \
                test_kmeans_loss, total_test_loss

        test_cluster_labels = infer_cluster_label(test_distances)
        print_cluster_metrics(testset.target, test_cluster_labels, "Test", write_files, logfile)
        test_ae, test_kmeans = sum(test_ae_losses) / n_test_batches, sum(test_kmeans_losses) / n_test_batches
        log(f"Test loss: {test_ae + lambda_ * test_kmeans}", write_files, logfile)
        log(f"Test auto encoder loss: {test_ae}", write_files, logfile)
        log(f"Test kmeans loss: {test_kmeans}", write_files, logfile)
        log("", write_files, logfile)
