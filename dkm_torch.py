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
from torch.optim import Adam
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from matplotlib.colors import TABLEAU_COLORS

from utils import cluster_acc
from ptb_img_utils import reconstruct_image, autoencoder_loss
from Modules import FCAutoencoder, DeepKMeans


def print_cluster_metrics(ground_truths, cluster_labels, phase, write_files, logfile):
    """
    @param phase: The phase of the experiment, either train or test.
    """
    acc = cluster_acc(ground_truths, cluster_labels)
    log(f"{phase} ACC: {acc}", write_files, logfile)
    ari = adjusted_rand_score(ground_truths, cluster_labels)
    log(f"{phase} ARI: {ari}", write_files, logfile)
    nmi = normalized_mutual_info_score(ground_truths, cluster_labels)
    log(f"{phase} NMI: {nmi}", write_files, logfile)


def infer_cluster_label(distances, data_size):
    cluster_labels = torch.zeros(data_size, dtype=torch.float)
    for i in range(data_size):
        cluster_index = torch.argmin(distances[:, i])
        cluster_labels[i] = cluster_index
    return cluster_labels


def log(content, to_disc, path):
    print(content)
    if to_disc:
        with path.open("a+") as f:
            f.write(f"{content}\n")


parser = argparse.ArgumentParser(description="Deep k-means algorithm")
parser.add_argument("-d", "--dataset", type=str.upper,
                    help="Dataset on which DKM will be run (one of USPS, MNIST, 20NEWS, RCV1)", required=True)
parser.add_argument("-v", "--validation", help="Split data into validation and test sets", action='store_true')
parser.add_argument("-p", "--pretrain", help="Pretrain the autoencoder and cluster representatives",
                    action='store_true')
parser.add_argument("-a", "--annealing",
                    help="Use an annealing scheme for the values of alpha (otherwise a constant is used)",
                    action='store_true')
parser.add_argument("-s", "--seeded", help="Use a fixed seed, different for each run", action='store_true')
parser.add_argument("-c", "--cpu", help="Force the program to run on CPU", action='store_true')
parser.add_argument("-l", "--lambda", type=float, default=1.0, dest="lambda_",
                    help="Value of the hyperparameter weighing the clustering loss against the reconstruction loss")
parser.add_argument("-e", "--p_epochs", type=int, default=50, help="Number of pretraining epochs")
parser.add_argument("-f", "--f_epochs", type=int, default=5, help="Number of fine-tuning epochs per alpha value")
parser.add_argument("-b", "--batch_size", type=int, default=256, help="Size of the minibatches used by the optimizer")
args = parser.parse_args()

# Dataset setting from arguments
if args.dataset == "USPS":
    import usps_specs as specs
elif args.dataset == "MNIST":
    import mnist_specs as specs
elif args.dataset == "20NEWS":
    import _20news_specs as specs
elif args.dataset == "RCV1":
    import rcv1_specs as specs
elif args.dataset == "PTB":
    import ptb_img_specs as specs
elif args.dataset == "PTBMAT":
    import ptb_matrix_specs as specs
else:
    parser.error("Unknown dataset!")
    exit()

# Parameter setting from arguments
n_pretrain_epochs = args.p_epochs
n_finetuning_epochs = args.f_epochs
lambda_ = args.lambda_
batch_size = args.batch_size # Size of the mini-batches used in the stochastic optimizer
n_batches = int(math.ceil(specs.n_samples / batch_size)) # Number of mini-batches
validation = args.validation # Specify if data should be split into validation and test sets
pretrain = args.pretrain # Specify if DKM's autoencoder should be pretrained
annealing = args.annealing # Specify if annealing should be used
seeded = args.seeded # Specify if runs are seeded
test_size = batch_size
write_files = True

now = datetime.now()
time_format = "%Y_%m_%dT%H_%M"
experiment_id = f"{args.dataset}_e_{n_pretrain_epochs}_f_{n_finetuning_epochs}_bs_{batch_size}_" \
                f"{now.strftime(time_format)}"
directory = Path.cwd() / "metrics" / experiment_id
directory.mkdir()
logfile = directory / f"log_{experiment_id}.txt"

log(f"Hyperparameters: lambda={lambda_}, pretrain_epochs={n_pretrain_epochs}, finetune_epochs={n_finetuning_epochs}, "
    f"batch_size={batch_size}", write_files, logfile)

# Hardware specifications
device = "cuda" if torch.cuda.is_available() else "cpu"
if args.cpu:
    device = "cpu"

# Definition of the randomly-drawn (0-10000) seeds to be used for each run
seeds = [8905, 9129, 291, 4012, 1256, 6819, 4678, 6971, 1362, 575]

n_runs = 1
for run in range(n_runs):
    log(f"Run {run}", write_files, logfile)

    if seeded:
        torch.manual_seed(seeds[run])
        np.random.seed(seeds[run])

    trainset, testset = specs.trainset, specs.testset
    autoencoder = FCAutoencoder(dimensions=specs.dimensions)
    autoencoder.to(device)
    optimizer = Adam(autoencoder.parameters(), 0.001)
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

        autoencoder.eval()
        test_indices = np.random.choice(range(len(testset)), size=test_size)
        test_batch = testset[test_indices][1].to(device)
        test_embeddings, test_reconstruction = autoencoder.forward(test_batch)
        test_embeddings, test_reconstruction = test_embeddings.detach().cpu(), test_reconstruction.detach().cpu()
        test_batch = test_batch.detach().cpu()

        kmeans_model = KMeans(n_clusters=specs.n_clusters, init="k-means++").fit(train_embeddings)
        train_cluster_labels = torch.from_numpy(kmeans_model.labels_).long()
        test_cluster_labels = torch.from_numpy(kmeans_model.predict(test_embeddings))

        log(f"Train auto encoder loss: {sum(ae_losses) / n_batches}", write_files, logfile)
        print_cluster_metrics(trainset.target, train_cluster_labels, "Train", write_files, logfile)
        log(f"Test auto encoder loss: {autoencoder_loss(test_batch, test_reconstruction)}", write_files, logfile)
        print_cluster_metrics(testset.target[test_indices], test_cluster_labels, "Test", write_files, logfile)
        log("", write_files, logfile)

    # Visualize auto encoder input and respective output
    img_shape = [specs.img_height, specs.img_width]
    now = datetime.now()
    time_format = "%Y_%m_%dT%H_%M"
    index = np.random.choice(len(testset), size=1)
    rand_input = testset[index][1].to(device)
    reconstruction = autoencoder.forward(rand_input)[1].cpu().detach().numpy()
    rand_input = rand_input.cpu().detach().numpy()
    input_img = reconstruct_image(rand_input, img_shape).convert("L")
    output_img = reconstruct_image(reconstruction, img_shape, transform=True).convert("L")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(input_img)
    ax1.set_title("input")
    ax2.imshow(output_img)
    ax2.set_title("reconstruction")
    if write_files:
        fig.savefig(directory / f"input_e{n_pretrain_epochs}_bs{batch_size}_{now.strftime(time_format)}.jpg")

    # Visualize embedding space using t-SNE
    centers = torch.from_numpy(kmeans_model.cluster_centers_)
    test_embeds_and_centers = torch.cat((test_embeddings, centers))
    fitted = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(test_embeds_and_centers)
    mapped_embeds, mapped_centers = fitted[:test_size, :], fitted[test_size:, :]
    colors = [*TABLEAU_COLORS.values()]
    color_map = {i: colors[i] for i in range(specs.embedding_size)}
    color_labels = [color_map[t.item()] for t in test_cluster_labels]
    fig2, ax3 = plt.subplots()
    for i in range(test_size):
        ax3.plot(mapped_embeds[i, 0], mapped_embeds[i, 1], ".", color=color_labels[i])
    for j in range(len(mapped_centers)):
        ax3.plot(mapped_centers[j, 0], mapped_centers[j, 1], "o", color=colors[j], label=f"{j}")
    ax3.legend(loc="upper right")
    if write_files:
        fig2.savefig(directory / f"torch_centers{now.strftime(time_format)}.jpg")

    # Train the full DKM model
    train_distances = torch.zeros((specs.n_clusters, len(trainset)))
    cluster_centers = torch.from_numpy(kmeans_model.cluster_centers_).to(device)
    cluster_centers.requires_grad = True
    deepkmeans = DeepKMeans(autoencoder, cluster_centers)
    optimizer = Adam(deepkmeans.parameters(), 0.001)
    for epoch in range(n_finetuning_epochs):
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        train_kmeans_losses, train_ae_losses = [], []

        deepkmeans.train()
        log(f"Training step: epoch {epoch}", write_files, logfile)
        for train_indices, train_batch, _ in train_loader:
            train_batch = train_batch.to(device)

            # Train one mini-batch
            optimizer.zero_grad()
            train_kmeans_distance_, stack_dist, train_reconstruction = deepkmeans.forward(train_batch)
            train_ae_loss_ = autoencoder_loss(train_batch, train_reconstruction)
            train_kmeans_loss_ = train_kmeans_distance_.sum(dim=0).mean()
            total_train_loss_ = train_ae_loss_ + train_kmeans_loss_
            total_train_loss_.backward()
            optimizer.step()

            train_kmeans_losses.append(train_kmeans_loss_.cpu())
            train_ae_losses.append(train_ae_loss_.cpu())
            train_distances[:, train_indices] = stack_dist.detach().cpu().float()

        train_cluster_labels = infer_cluster_label(train_distances, len(trainset)).cpu().long()
        print_cluster_metrics(trainset.target, train_cluster_labels, "Train", write_files, logfile)
        train_ae, train_kmeans = sum(train_ae_losses) / n_batches, sum(train_kmeans_losses) / n_batches
        log(f"Train loss: {train_ae + train_kmeans}", write_files, logfile)
        log(f"Train auto encoder loss: {train_ae}", write_files, logfile)
        log(f"Train kmeans loss: {train_kmeans}", write_files, logfile)

        deepkmeans.eval()
        test_indices = np.random.choice(range(len(testset)), size=test_size)
        test_batch = testset[test_indices][1].to(device)
        test_kmeans_distance, test_stack_dist, test_reconstruction = deepkmeans.forward(test_batch)
        test_ae_loss = autoencoder_loss(test_batch, test_reconstruction).item()
        test_kmeans_loss = test_kmeans_distance.sum(dim=0).mean().item()
        total_test_loss = test_ae_loss + test_kmeans_loss
        test_cluster_labels = infer_cluster_label(test_stack_dist, test_size).cpu().long()
        print_cluster_metrics(testset.target[test_indices], test_cluster_labels, "Test", write_files, logfile)
        log(f"Test loss: {total_test_loss}", write_files, logfile)
        log(f"Test auto encoder loss: {test_ae_loss}", write_files, logfile)
        log(f"Test kmeans loss: {test_kmeans_loss}", write_files, logfile)
        log("", write_files, logfile)
