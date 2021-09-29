#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

import os
import math
from pathlib import Path
from datetime import datetime
import numpy as np
import argparse

import torch.cuda
from torch.optim import Adam
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from utils import cluster_acc
from utils import next_batch
from ptb_img_utils import reconstruct_image, get_train_and_test_loaders, autoencoder_loss
from Modules import FCAutoencoder, DeepKMeans

# Delete this import
from ptb_img_utils import PTBImgSet, DataLoader


def print_cluster_metrics(ground_truths, cluster_labels, phase):
    """
    @param phase: The phase of the experiment, either train or test.
    """
    acc = cluster_acc(ground_truths, cluster_labels)
    print(f"{phase} ACC: {acc}")
    ari = adjusted_rand_score(ground_truths, cluster_labels)
    print(f"{phase} ARI: {ari}")
    nmi = normalized_mutual_info_score(ground_truths, cluster_labels)
    print(f"{phase} NMI: {nmi}", flush=True)


def infer_cluster_label(distances, data_size):
    cluster_labels = np.zeros(data_size, dtype=float)
    for i in range(data_size):
        cluster_index = np.argmin(distances[:, i])
        cluster_labels[i] = cluster_index
    return cluster_labels.astype(np.int64)


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

print(f"Hyperparameters: lambda={lambda_}, pretrain_epochs={n_pretrain_epochs}, finetune_epochs="
      f"{n_finetuning_epochs}, batch_size={batch_size}")

# Define the alpha scheme depending on if the approach includes annealing/pretraining
if annealing and not pretrain:
    constant_value = 1  # specs.embedding_size # Used to modify the range of the alpha scheme
    max_n = 40  # Number of alpha values to consider
    alphas = np.zeros(max_n, dtype=float)
    alphas[0] = 0.1
    for i in range(1, max_n):
        alphas[i] = (2 ** (1 / (np.log(i + 1)) ** 2)) * alphas[i - 1]
    alphas = alphas / constant_value
elif not annealing and pretrain:
    constant_value = 1  # specs.embedding_size # Used to modify the range of the alpha scheme
    max_n = 1  # Number of alpha values to consider (constant values are used here)
    alphas = 1000*np.ones(max_n, dtype=float) # alpha is constant
    alphas = alphas / constant_value
else:
    parser.error("Run with either annealing (-a) or pretraining (-p), but not both.")
    exit()


# Dataset on which the computation graph will be run
data = specs.data
target = specs.target

# Select only the labels which are to be used in the evaluation (disjoint for validation and test)
# train_target = target[specs.train_indices]
# train_data = data[specs.train_indices]
# test_target = target[specs.test_indices]
# test_data = data[specs.test_indices]

train_target = target
train_data = data
test_target = target
test_data = data
# train_iter, test_iter = get_train_and_test_iterators(Path.cwd() / "split" / "ptb-img" / "3k", 256, 100)

# Hardware specifications
device = "cuda" if torch.cuda.is_available() else "cpu"
if args.cpu:
    device = "cpu"
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Run on CPU instead of GPU if batch_size is small

# Definition of the randomly-drawn (0-10000) seeds to be used for each run
seeds = [8905, 9129, 291, 4012, 1256, 6819, 4678, 6971, 1362, 575]

n_runs = 1
for run in range(n_runs):
    print("Run", run)

    dataset = PTBImgSet(data, target)
    autoencoder = FCAutoencoder(dimensions=specs.dimensions)
    autoencoder.to(device)
    optimizer = Adam(autoencoder.parameters(), 0.001)
    loss = autoencoder_loss
    kmeans_model = None
    if pretrain:
        print("Starting autoencoder pretraining...")
        autoencoder.train()

        for epoch in range(n_pretrain_epochs):
            print(f"Pretraining step: epoch {epoch}", flush=True)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            train_embeddings = np.zeros((train_data.shape[0], specs.embedding_size), dtype=float)
            test_embeddings = np.zeros((test_data.shape[0], specs.embedding_size), dtype=float)
            ae_losses = []
            for train_indices, train_batch, _ in train_loader:
                train_batch = train_batch.to(device)

                # Train one mini-batch
                autoencoder.zero_grad()
                train_embedding_, reconstruction = autoencoder.forward(train_batch)
                train_ae_loss_ = loss(train_batch, reconstruction)
                train_ae_loss_.backward()
                optimizer.step()

                # Save metrics to cpu memory
                train_embedding_ = train_embedding_.cpu().detach().numpy()
                ae_losses.append(train_ae_loss_.item())
                for j in range(len(train_embedding_)):
                    train_embeddings[train_indices[j], :] = train_embedding_[j, :]
                    # test_embeddings[test_indices[j], :] = test_embedding_[j, :]

            kmeans_model = KMeans(n_clusters=specs.n_clusters, init="k-means++").fit(train_embeddings)
            train_cluster_labels = kmeans_model.labels_
            # test_cluster_labels = kmeans_model.predict(test_nonzeros)

            print(f"Train auto encoder loss: {sum(ae_losses) / n_batches}")
            print_cluster_metrics(target, train_cluster_labels, "Train")
            # print_cluster_metrics(test_ground_truths, test_cluster_labels, "Test")
            print("", flush=True)


        # The cluster centers are used to initialize the cluster representatives in DKM
        # sess.run(tf.compat.v1.assign(cg.cluster_rep, kmeans_model.cluster_centers_))
        # Visualize auto encoder input and respective output
        autoencoder.eval()
        if args.dataset == "PTB":
            img_shape = [314, 384]
            now = datetime.now()
            random_indices = np.random.choice(data.shape[0], size=3)
            rand_input = data[random_indices, :]
            _, reconstruction = autoencoder.forward(rand_input)
            for i in range(len(random_indices)):
                input_img = reconstruct_image(rand_input[i, :], img_shape)
                output_img = reconstruct_image(reconstruction[i, :], img_shape)
                input_img.save(Path.cwd() / "metrics" / f"input{i}_e{n_pretrain_epochs}_bs{batch_size}"
                                                        f"_{now.isoformat()}")
                output_img.save(Path.cwd() / "metrics" / f"output{i}_e{n_pretrain_epochs}_bs{batch_size}"
                                                         f"_{now.isoformat()}")

    # Variables to save tensor content
    train_distances = np.zeros((specs.n_clusters, train_data.shape[0]))
    test_distances = np.zeros((specs.n_clusters, test_data.shape[0]))

    # Train the full DKM model
    cluster_centers = torch.from_numpy(kmeans_model.cluster_centers_).to(device)
    deepkmeans = DeepKMeans(autoencoder, cluster_centers)
    deepkmeans.train()
    optimizer = Adam(deepkmeans.parameters(), 0.001)
    for epoch in range(n_finetuning_epochs):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_kmeans_losses, train_ae_losses = [], []
        print(f"Training step: epoch {epoch}", flush=True)
        for train_indices, train_batch, _ in train_loader:
            train_batch = train_batch.to(device)

            # Train one mini-batch
            deepkmeans.zero_grad()
            train_kmeans_distance_, train_reconstruction = deepkmeans.forward(train_batch)
            train_ae_loss_ = autoencoder_loss(train_batch, train_reconstruction)
            train_kmeans_loss_ = train_kmeans_distance_.sum(dim=0).mean()
            total_train_loss_ = train_ae_loss_ + train_kmeans_loss_
            total_train_loss_.backward()
            optimizer.step()

            # Save metrics to cpu memory
            train_ae_loss_ = train_ae_loss_.cpu().detach().numpy()
            train_kmeans_loss_ = train_kmeans_loss_.cpu().detach().numpy()
            train_kmeans_distance_ = train_kmeans_distance_.cpu().detach().numpy()
            train_kmeans_losses.append(train_kmeans_loss_)
            train_ae_losses.append(train_ae_loss_)
            for j in range(len(train_indices)):
                train_distances[:, train_indices[j]] = train_kmeans_distance_[:, j]
                # test_distances[:, test_indices[j]] = test_stack_dist_[:, j]


        train_cluster_labels = infer_cluster_label(train_distances, train_data.shape[0])
        # test_cluster_labels = infer_cluster_label(test_distances, test_data.shape[0])
        print_cluster_metrics(target, train_cluster_labels, "Train")
        # Evaluate the clustering performance every print_val alpha and for last alpha
        train_ae, train_kmeans = sum(train_ae_losses) / n_batches, sum(train_kmeans_losses) / n_batches
        print(f"Train loss: {train_ae + train_kmeans}")
        print(f"Train auto encoder loss: {train_ae}")
        print(f"Train kmeans loss: {train_kmeans}")

        # print_cluster_metrics(target, test_cluster_labels, "Test")
        # print("Test loss: ", test_loss_)
        # print("Test auto encoder loss: ", test_ae_loss_)
        # print("Test kmeans loss: ", test_kmeans_loss_)
        # print("")
