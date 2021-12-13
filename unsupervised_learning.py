#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

from datetime import datetime

import numpy as np
import torch.cuda
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torch
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from utils import cluster_acc, map_clusterlabels_to_groundtruth, get_color_map, conv_regularization_loss
from ptb_img_utils import reconstruct_image
from modules.misc import DeepKMeans
from modules.autoencoders import get_autoencoder
from learning_super import Learner
from losses import sparsity_loss


class UnsupervisedLearner(Learner):

    def __init__(self, args, specs, autoencoder_loss):
        super().__init__(args, specs)
        self.autoencoder_loss = autoencoder_loss

    def run_repeated_learning(self):
        for _ in range(self.n_runs):
            self._log(f"Run {self.run}")
            self.autoencoder = get_autoencoder(
                self.autoencoder_name,
                self.img_height,
                self.img_width,
                self.embedding_size,
                self.n_channels,
            ).to(self.device)
            if self.seeded:
                torch.manual_seed(self.seeds[self.run])
                np.random.seed(self.seeds[self.run])
            self.pretrain()
            self.finetune()
            self.run += 1
        duration = datetime.now() - self.start_time
        self._log(f"Learning duration: {duration}")

    def pretrain(self):
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), self.lr)

        for epoch in range(self.n_pretrain_epochs):
            self._log(f"Pretraining step: epoch {epoch}")
            train_loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
            train_embeddings = torch.zeros((len(self.trainset), self.embedding_size), dtype=torch.float)
            ae_losses = []
            self.autoencoder.train()
            for train_indices, train_batch, _ in train_loader:
                train_batch = train_batch.to(self.device)

                # Train one mini-batch
                self.autoencoder.zero_grad(set_to_none=True)
                train_embedding_, reconstruction = self.autoencoder.forward(train_batch)
                train_ae_loss_ = self.autoencoder_loss(reconstruction, train_batch)
                # conv_regular = conv_regularization_loss(self.autoencoder)
                total_loss = train_ae_loss_ #+ 0.2*sparsity_loss(train_embedding_, 0.05)     # 0*conv_regular
                total_loss.backward()
                # train_ae_loss_.backward()
                torch.nn.utils.clip_grad_value_(self.autoencoder.parameters(), 10.0)
                optimizer.step()

                # Save metrics to cpu memory
                ae_losses.append(train_ae_loss_.item())
                train_embeddings[train_indices, :] = train_embedding_.detach().cpu()

            del train_embedding_, reconstruction, train_ae_loss_, train_batch

            test_loader = DataLoader(self.testset, batch_size=self.test_size, shuffle=True, pin_memory=True)
            test_embeddings = torch.zeros((len(self.testset), self.embedding_size), dtype=torch.float)
            test_ae_losses = []
            self.autoencoder.eval()
            with torch.no_grad():
                for test_indices, test_batch, _ in test_loader:
                    test_batch = test_batch.to(self.device)

                    # Eval one mini-batch
                    test_embedding_, test_reconstruction = self.autoencoder.forward(test_batch)
                    test_ae_loss = self.autoencoder_loss(test_reconstruction, test_batch)

                    # Save metrics to cpu memory
                    test_ae_losses.append(test_ae_loss.item())
                    test_embeddings[test_indices, :] = test_embedding_.detach().cpu()

            self.kmeans_model = KMeans(n_clusters=self.n_clusters, init="k-means++").fit(train_embeddings)
            train_cluster_labels = torch.from_numpy(self.kmeans_model.labels_).long()
            test_cluster_labels = torch.from_numpy(self.kmeans_model.predict(test_embeddings))

            self._log(f"Train auto encoder loss: {sum(ae_losses) / self.n_batches}")
            self._print_cluster_metrics(self.trainset.target, train_cluster_labels, "Train")
            self._log(f"Test auto encoder loss: {sum(test_ae_losses) / self.n_test_batches}")
            self._print_cluster_metrics(self.testset.target, test_cluster_labels, "Test")
            self._log("")

        if self.is_writing_to_disc:
            self._visualize_autoencoder()
            self._cluster_plot(test_embeddings, "pretrain")

    def finetune(self):
        # Train the full DKM model
        cluster_centers = torch.from_numpy(self.kmeans_model.cluster_centers_).to(self.device)
        deep_cluster_net = DeepKMeans(self.autoencoder, cluster_centers)
        optimizer = torch.optim.Adam(deep_cluster_net.parameters(), self.lr)
        for epoch in range(self.n_finetuning_epochs):
            train_distances = torch.zeros((len(self.trainset), self.n_clusters))
            train_loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
            train_cluster_losses, train_ae_losses = [], []

            deep_cluster_net.train()
            self._log(f"Training step: epoch {epoch}")
            for train_indices, train_batch, _ in train_loader:
                train_batch = train_batch.to(self.device)

                # Train one mini-batch
                deep_cluster_net.zero_grad(set_to_none=True)
                train_cluster_distance, train_batch_distances, train_reconstruction, train_embedding = \
                    deep_cluster_net.forward(
                    train_batch)
                train_ae_loss_ = self.autoencoder_loss(train_reconstruction, train_batch)
                train_cluster_loss = train_cluster_distance.sum(dim=1).mean()
                # conv_regular = conv_regularization_loss(self.autoencoder)
                total_train_loss_ = train_ae_loss_ + self.lambda_*train_cluster_loss\
                                                   + 0.2*sparsity_loss(train_embedding, 0.05)       # + 0*conv_regular
                total_train_loss_.backward()
                optimizer.step()

                # Save metrics do cpu memory
                train_cluster_losses.append(train_cluster_loss.item())
                train_ae_losses.append(train_ae_loss_.item())
                train_distances[train_indices, :] = train_batch_distances.detach().cpu().float()

            del train_batch, train_cluster_distance, train_batch_distances, train_reconstruction, train_ae_loss_, \
                train_cluster_loss, total_train_loss_

            self._print_finetune_metrics(train_distances, train_ae_losses, train_cluster_losses, "Train")

            test_distances = torch.zeros((len(self.testset), self.n_clusters))
            test_embeddings = torch.zeros((len(self.testset), self.embedding_size))
            test_loader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=True)
            test_cluster_losses, test_ae_losses = [], []
            deep_cluster_net.eval()
            with torch.no_grad():
                for test_indices, test_batch, _ in test_loader:
                    test_batch = test_batch.to(self.device)

                    # Eval one mini-batch
                    test_kmeans_distance, test_batch_distances, test_reconstruction, test_embedding_ = deep_cluster_net\
                        .forward(test_batch)
                    test_ae_loss = self.autoencoder_loss(test_reconstruction, test_batch).item()
                    test_cluster_loss = test_kmeans_distance.sum(dim=1).mean().item()

                    # Save metrics to cpu memory
                    test_ae_losses.append(test_ae_loss)
                    test_cluster_losses.append(test_cluster_loss)
                    test_distances[test_indices, :] = test_batch_distances.detach().cpu().float()
                    test_embeddings[test_indices, :] = test_embedding_.detach().cpu()

                del test_batch, test_kmeans_distance, test_batch_distances, test_reconstruction, test_ae_loss, \
                    test_cluster_loss

            self._print_finetune_metrics(test_distances, test_ae_losses, test_cluster_losses, "Test")
            self._log("")

        # Visualize embedding space using t-SNE
        if self.is_writing_to_disc:
            self._cluster_plot(test_embeddings, "finetune")

    def _cluster_plot(self, test_embeds, phase):
        mapped_embeds = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(test_embeds)
        color_map = get_color_map(self.n_clusters)
        color_labels = [color_map[t.item()] for t in self.testset.target]
        fig2, ax3 = plt.subplots()
        for i in range(len(self.testset)):
            if self.dataset_name == "MNIST":
                ax3.plot(mapped_embeds[i, 0], mapped_embeds[i, 1], ".", color=color_labels[i], label=f"{self.testset.target[i]}")
            else:
                ax3.plot(mapped_embeds[i, 0], mapped_embeds[i, 1], ".", color=color_labels[i],
                         label=f"{self.inverse_disease_mapping[self.testset.target[i].item()]}")
        self._legend_without_duplicate_labels(ax3, loc="upper right")
        fig2.savefig(self.directory / f"torch_centers_run{self.run}_{phase}.jpg")

    @staticmethod
    def _legend_without_duplicate_labels(ax, **kwargs):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique), **kwargs)

    def _visualize_autoencoder(self):
        img_shape = [self.img_height, self.img_width]
        now = datetime.now()
        time_format = "%Y_%m_%dT%H_%M"
        index = np.random.choice(len(self.testset), size=1)
        rand_input = self.testset[index][1].to(self.device)
        with torch.no_grad():
            reconstruction = self.autoencoder.forward(rand_input)[1].cpu().detach().numpy()
        rand_input = rand_input.cpu().detach().numpy()
        input_img = reconstruct_image(rand_input, img_shape).convert("L")
        output_img = reconstruct_image(reconstruction, img_shape, transform=True).convert("L")
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(input_img)
        ax1.set_title("input")
        ax2.imshow(output_img)
        ax2.set_title("reconstruction")
        fig.savefig(self.directory / f"input_e{self.n_pretrain_epochs}_bs{self.batch_size}_{now.strftime(time_format)}.jpg")

    def _print_finetune_metrics(self, distances, ae_losses, kmeans_losses, phase):
        cluster_labels = self.infer_cluster_label(distances)
        if phase == "Train":
            self._print_cluster_metrics(self.trainset.target, cluster_labels, phase)
        else:
            self._print_cluster_metrics(self.testset.target, cluster_labels, phase)
        avg_ae_loss, avg_cluster_loss = sum(ae_losses) / len(ae_losses), sum(kmeans_losses) / len(kmeans_losses)
        self._log(f"{phase} loss: {avg_ae_loss + self.lambda_ * avg_cluster_loss}")
        self._log(f"{phase} auto encoder loss: {avg_ae_loss}")
        self._log(f"{phase} kmeans loss: {avg_cluster_loss}")

    def _print_cluster_metrics(self, ground_truths, cluster_labels, phase):
        """
        @param phase: The phase of the experiment, either train or test.
        """
        predictions = map_clusterlabels_to_groundtruth(ground_truths, cluster_labels)
        acc = cluster_acc(ground_truths, predictions)
        self._log(f"{phase} ACC: {acc}")
        ari = adjusted_rand_score(ground_truths, predictions)
        self._log(f"{phase} ARI: {ari}")
        nmi = normalized_mutual_info_score(ground_truths, predictions)
        self._log(f"{phase} NMI: {nmi}")

    @staticmethod
    def infer_cluster_label(distances):
        return torch.argmin(distances, dim=1)
