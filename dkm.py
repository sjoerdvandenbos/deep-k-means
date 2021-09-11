#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

import os
import math
import numpy as np
import argparse
import tensorflow as tf
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from utils import cluster_acc
from compgraph import DkmCompGraph


def print_cluster_metrics(ground_truths, cluster_labels, phase):
    """
    @param phase: The phase of the experiment, either train or test.
    """
    acc = cluster_acc(ground_truths, cluster_labels)
    print(f"{phase} ACC: {acc}")
    ari = adjusted_rand_score(ground_truths, cluster_labels)
    print(f"{phase} ARI: {ari}")
    nmi = normalized_mutual_info_score(ground_truths, cluster_labels)
    print(f"{phase} NMI: {nmi}")


def infer_cluster_label(distances, data_size):
    cluster_labels = np.zeros((data_size), dtype=float)
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

if args.dataset == "PTB":
    from ptb_img_utils import next_batch
else:
    from utils import next_batch

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
elif args.dataset == "PTB_MAT":
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

print("Hyperparameters...")
print("lambda =", lambda_)

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
train_target = target[specs.train_indices]
train_data = data[specs.train_indices]
test_target = target[specs.test_indices]
test_data = data[specs.test_indices]

# Hardware specifications
if args.cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Run on CPU instead of GPU if batch_size is small
config = tf.compat.v1.ConfigProto()

# Definition of the randomly-drawn (0-10000) seeds to be used for each run
seeds = [8905, 9129, 291, 4012, 1256, 6819, 4678, 6971, 1362, 575]

n_runs = 1
for run in range(n_runs):
    tf.compat.v1.reset_default_graph()
    # Use a fixed seed for this run, as defined in the seed list
    if seeded:
        tf.compat.v1.set_random_seed(seeds[run])
        np.random.seed(seeds[run])

    print("Run", run)

    # Define the computation graph for DKM
    cg = DkmCompGraph([specs.dimensions, specs.activations, specs.names], specs.n_clusters, lambda_)

    # Run the computation graph
    with tf.compat.v1.Session(config=config) as sess:
        # Initialization
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        # Pretrain if specified
        if pretrain:
            print("Starting autoencoder pretraining...")

            # Variables to save pretraining tensor content
            train_embeddings = np.zeros((train_data.shape[0], specs.embedding_size), dtype=float)
            test_embeddings = np.zeros((test_data.shape[0], specs.embedding_size), dtype=float)

            # First, pretrain the autoencoder
            for epoch in range(n_pretrain_epochs):
                print(f"Pretraining step: epoch {epoch}", flush=True)

                for b in range(n_batches):
                    print(f"Starting batch {b} out of {n_batches} batches.", flush=True)
                    # Fetch a random data batch of the specified size
                    train_indices, train_batch = next_batch(batch_size, train_data)

                    # Run the computation graph until pretrain_op (only on autoencoder) on the data batch
                    _, train_embedding_, ae_loss_, stack_dist = sess.run((cg.pretrain_op, cg.embedding, cg.ae_loss, cg.stack_dist),
                                                        feed_dict={cg.input: train_batch})
                    del train_batch, _

                    test_indices, test_batch = next_batch(batch_size, test_data)
                    test_embedding_ = sess.run(cg.embedding, feed_dict={cg.input: test_batch})

                    # Save the embeddings for batch samples
                    for j in range(batch_size):
                        train_embeddings[train_indices[j], :] = train_embedding_[j, :]
                        test_embeddings[test_indices[j], :] = test_embedding_[j, :]

                nonzero_train_indices = np.all(train_embeddings != 0, axis=1)
                train_nonzeros = train_embeddings[nonzero_train_indices]
                train_ground_truths = train_target[nonzero_train_indices]

                nonzero_test_indices = np.all(test_embeddings != 0, axis=1)
                test_nonzeros = test_embeddings[nonzero_test_indices]
                test_ground_truths = test_target[nonzero_test_indices]

                kmeans_model = KMeans(n_clusters=specs.n_clusters, init="k-means++").fit(train_nonzeros)
                train_cluster_labels = kmeans_model.labels_
                test_cluster_labels = kmeans_model.predict(test_nonzeros)

                print(f"Auto encoder loss: {ae_loss_}")
                print_cluster_metrics(train_ground_truths, train_cluster_labels, "Train")
                print_cluster_metrics(test_ground_truths, test_cluster_labels, "Test")
                print("", flush=True)


            # The cluster centers are used to initialize the cluster representatives in DKM
            sess.run(tf.compat.v1.assign(cg.cluster_rep, kmeans_model.cluster_centers_))

        # Variables to save tensor content
        train_distances = np.zeros((specs.n_clusters, train_data.shape[0]))
        test_distances = np.zeros((specs.n_clusters, test_data.shape[0]))

        list_train_acc = []
        list_train_ari = []
        list_train_nmi = []
        list_test_acc = []
        list_test_ari = []
        list_test_nmi = []

        # Train the full DKM model
        for epoch in range(n_finetuning_epochs):
            print(f"Training step: epoch {epoch}", flush=True)
            # Loop over the samples
            for _ in range(n_batches):
                # Fetch a random data batch of the specified size
                train_indices, train_batch = next_batch(batch_size, train_data)
                # Run the computation graph on the data batch
                _, train_loss_, train_stack_dist_, cluster_rep_, train_ae_loss_, train_kmeans_loss_ =\
                    sess.run((cg.train_op, cg.loss, cg.stack_dist, cg.cluster_rep, cg.ae_loss, cg.kmeans_loss),
                             feed_dict={cg.input: train_batch, cg.alpha: alphas[0]})
                del train_batch, _

                test_indices, test_batch = next_batch(batch_size, test_data)
                test_loss_, test_stack_dist_, test_ae_loss, test_kmeans_loss_ =\
                    sess.run((cg.loss, cg.stack_dist, cg.ae_loss, cg.kmeans_loss),
                             feed_dict={cg.input: test_batch, cg.alpha: alphas[0]})

                # Save the distances for batch samples
                for j in range(batch_size):
                    train_distances[:, train_indices[j]] = train_stack_dist_[:, j]
                    test_distances[:, test_indices[j]] = test_stack_dist_[:, j]

            # Evaluate the clustering performance every print_val alpha and for last alpha
            print("train loss:", train_loss_)
            print("train auto encoder loss:", ae_loss_)
            print("train kmeans loss:", train_kmeans_loss_)

            train_cluster_labels = infer_cluster_label(train_distances, train_data.shape[0])
            test_cluster_labels = infer_cluster_label(test_distances, test_data.shape[0])

            # Here we want to get rid of the embeddings that were not randomly picked by the next_batch()
            nonzero_train_indices = np.all(train_distances != 0, axis=0)
            nonzero_train_cluster_labels = train_cluster_labels[nonzero_train_indices]
            nonzero_train_ground_truths = train_target[nonzero_train_indices]
            print_cluster_metrics(nonzero_train_ground_truths, nonzero_train_cluster_labels, "Train")

            nonzero_test_indices = np.all(test_distances != 0, axis=0)  # Col indices where all elements were 0
            nonzero_test_cluster_labels = test_cluster_labels[nonzero_test_indices]
            nonzero_test_ground_truths = test_target[nonzero_test_indices]
            print_cluster_metrics(nonzero_test_ground_truths, nonzero_test_cluster_labels, "Test")

# list_train_acc = np.array(list_train_acc)
# print("Average validation ACC: {:.3f} +/- {:.3f}".format(np.mean(list_train_acc), np.std(list_train_acc)))
# list_train_ari = np.array(list_train_ari)
# print("Average validation ARI: {:.3f} +/- {:.3f}".format(np.mean(list_train_ari), np.std(list_train_ari)))
# list_train_nmi = np.array(list_train_nmi)
# print("Average validation NMI: {:.3f} +/- {:.3f}".format(np.mean(list_train_nmi), np.std(list_train_nmi)))
#
# list_test_acc = np.array(list_test_acc)
# print("Average test ACC: {:.3f} +/- {:.3f}".format(np.mean(list_test_acc), np.std(list_test_acc)))
# list_test_ari = np.array(list_test_ari)
# print("Average test ARI: {:.3f} +/- {:.3f}".format(np.mean(list_test_ari), np.std(list_test_ari)))
# list_test_nmi = np.array(list_test_nmi)
# print("Average test NMI: {:.3f} +/- {:.3f}".format(np.mean(list_test_nmi), np.std(list_test_nmi)))
