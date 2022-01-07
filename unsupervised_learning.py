#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

from datetime import datetime

import numpy as np
import torch.cuda
from matplotlib import pyplot as plt
import torch
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from utils import cluster_acc, map_clusterlabels_to_groundtruth, get_color_map, get_clusterlabel_to_groundtruth_map
from ptb_img_utils import reconstruct_image
from modules.misc import DeepKMeans, PolarMapper
from modules.autoencoders import get_autoencoder
from modules.autoencoder_setups import get_ae_setup
from learning_super import Learner


class UnsupervisedLearner(Learner):

    def __init__(self, args, specs, autoencoder_setup):
        super().__init__(args, specs)
        self.autoencoder_setup = autoencoder_setup
        self.polar_mapping_enabled = args.polar_mapping
        self._log(f"polar_mapping_enabled={self.polar_mapping_enabled}")
        self._log(f"autoencder setup:\n{autoencoder_setup}")
        self.deep_cluster_net = None

    def run_repeated_learning(self):
        for _ in range(self.n_runs):
            self._log(f"Run {self.run}")
            self.autoencoder_setup.set_new_autoencoder()
            if self.is_writing_to_disc and self.run == 0:
                self.save_architecture()
            if self.seeded:
                torch.manual_seed(self.seeds[self.run])
                np.random.seed(self.seeds[self.run])
            self.pretrain()
            self.finetune()
            self.run += 1
        duration = datetime.now() - self.start_time
        self._log(f"Learning duration: {duration}")

    def pretrain(self):
        optimizer = torch.optim.Adam(self.autoencoder_setup.parameters(), self.lr)
        train_polarmapper = PolarMapper()
        test_polarmapper = PolarMapper()
        test_embeds = torch.zeros((len(self.testset), self.embedding_size), dtype=torch.float)
        train_embeds = torch.zeros((len(self.trainset), self.embedding_size), dtype=torch.float)

        for epoch in range(self.n_pretrain_epochs):
            self._log(f"Pretraining step: epoch {epoch}")
            train_loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
            train_polar_embeds = torch.zeros_like(train_embeds)
            if self.polar_mapping_enabled and epoch > 0:
                train_polarmapper.set_center(train_embeds.to(self.device))
            ae_losses = []
            self.autoencoder_setup.train()
            for train_indices, train_batch, _ in train_loader:
                train_batch = train_batch.to(self.device)

                # Train one mini-batch
                self.autoencoder_setup.zero_grad(set_to_none=True)
                train_embed_, reconstruction = self.autoencoder_setup.forward(train_batch)
                if self.polar_mapping_enabled:
                    train_polar_embed_ = train_polarmapper(train_embed_)
                train_ae_loss_ = self.autoencoder_setup.get_autoencoder_loss(reconstruction, train_batch)
                train_ae_loss_.backward()
                torch.nn.utils.clip_grad_value_(self.autoencoder_setup.parameters(), 10.0)
                optimizer.step()

                # Save metrics to cpu memory
                ae_losses.append(train_ae_loss_.item())
                train_embeds[train_indices, :] = train_embed_.detach().cpu()
                if self.polar_mapping_enabled:
                    train_polar_embeds[train_indices, :] = train_polar_embed_.detach().cpu()

            del train_embed_, reconstruction, train_ae_loss_, train_batch

            test_loader = DataLoader(self.testset, batch_size=self.test_size, shuffle=True, pin_memory=True)
            test_polar_embeddings = torch.zeros_like(test_embeds)
            test_ae_losses = []
            self.autoencoder_setup.eval()
            with torch.no_grad():
                if self.polar_mapping_enabled and epoch > 0:
                    test_polarmapper.set_center(test_embeds.to(self.device))
                for test_indices, test_batch, _ in test_loader:
                    test_batch = test_batch.to(self.device)

                    # Eval one mini-batch
                    test_embedding_, test_reconstruction = self.autoencoder_setup.forward(test_batch)
                    if self.polar_mapping_enabled:
                        test_polar_embedding_ = test_polarmapper(test_embedding_)
                    test_ae_loss_ = self.autoencoder_setup.get_autoencoder_loss(test_reconstruction, test_batch)

                    # Save metrics to cpu memory
                    test_ae_losses.append(test_ae_loss_.item())
                    test_embeds[test_indices, :] = test_embedding_.detach().cpu()
                    if self.polar_mapping_enabled:
                        test_polar_embeddings[test_indices, :] = test_polar_embedding_.detach().cpu()

            if self.polar_mapping_enabled:
                self.kmeans_model = KMeans(n_clusters=self.n_clusters).fit(train_polar_embeds)
                train_cluster_labels = torch.from_numpy(self.kmeans_model.labels_).long()
                test_cluster_labels = torch.from_numpy(self.kmeans_model.predict(test_polar_embeddings))
            else:
                self.kmeans_model = KMeans(n_clusters=self.n_clusters).fit(train_embeds)
                train_cluster_labels = torch.from_numpy(self.kmeans_model.labels_).long()
                test_cluster_labels = torch.from_numpy(self.kmeans_model.predict(test_embeds))

            self._log(f"Train auto encoder loss: {sum(ae_losses) / self.n_batches}")
            self._print_cluster_metrics(self.trainset.target, train_cluster_labels, "Train")
            self._log(f"Test auto encoder loss: {sum(test_ae_losses) / self.n_test_batches}")
            self._print_cluster_metrics(self.testset.target, test_cluster_labels, "Test")
            self._log("")

        if self.is_writing_to_disc:
            if self.dataset_name == "PTB" or self.dataset_name == "MNIST":
                self._visualize_autoencoder_image()
            elif self.dataset_name == "PTBMAT":
                self._visualize_autoencoder_matrix()
            if self.polar_mapping_enabled:
                self._cluster_plot(test_embeds, test_cluster_labels, "pretrain", show_centers=False)
                self._cluster_plot(test_polar_embeddings, test_cluster_labels, "pretrain_polar")
            else:
                self._cluster_plot(test_embeds, test_cluster_labels, "pretrain")

    def finetune(self):
        # Train the full DKM model
        cluster_centers = torch.from_numpy(self.kmeans_model.cluster_centers_).to(self.device)
        self.deep_cluster_net = DeepKMeans(self.autoencoder_setup, cluster_centers)
        optimizer = torch.optim.Adam(self.deep_cluster_net.parameters(), self.lr)
        train_embeds = torch.zeros((len(self.trainset), self.embedding_size), dtype=torch.float)
        test_embeds = torch.zeros((len(self.testset), self.embedding_size), dtype=torch.float)
        train_polar_embeds = torch.zeros_like(train_embeds)
        test_polar_embeds = torch.zeros_like(test_embeds)
        if self.polar_mapping_enabled:
            train_polarmapper = PolarMapper()
            test_polarmapper = PolarMapper()
        else:
            train_polarmapper = None
            test_polarmapper = None
        for epoch in range(self.n_finetuning_epochs):
            if self.polar_mapping_enabled and epoch > 0:
                train_polarmapper.set_center(train_embeds.to(self.device))
            train_distances = torch.zeros((len(self.trainset), self.n_clusters))
            train_loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
            train_cluster_losses, train_ae_losses = [], []

            self.deep_cluster_net.train()
            self._log(f"Training step: epoch {epoch}")
            for train_indices, train_batch, _ in train_loader:
                train_batch = train_batch.to(self.device)

                # Train one mini-batch
                self.deep_cluster_net.zero_grad(set_to_none=True)
                train_cluster_distance, train_batch_distances, train_reconstruction, train_embed_, \
                                        train_polar_embed_ = self.deep_cluster_net.forward(train_batch, train_polarmapper)
                train_ae_loss_ = self.deep_cluster_net.autoencoder.get_autoencoder_loss(train_reconstruction, train_batch)
                train_cluster_loss = train_cluster_distance.sum(dim=1).mean()
                total_train_loss_ = train_ae_loss_ + self.lambda_*train_cluster_loss
                total_train_loss_.backward()
                optimizer.step()

                # Save metrics to cpu memory
                train_cluster_losses.append(train_cluster_loss.item())
                train_ae_losses.append(train_ae_loss_.item())
                train_distances[train_indices, :] = train_batch_distances.detach().cpu().float()
                train_embeds[train_indices, :] = train_embed_.detach().cpu().float()
                if self.polar_mapping_enabled:
                    train_polar_embeds[train_indices, :] = train_polar_embed_.detach().cpu().float()

            del train_batch, train_cluster_distance, train_batch_distances, train_reconstruction, train_ae_loss_, \
                train_cluster_loss, total_train_loss_

            self._print_finetune_metrics(train_distances, train_ae_losses, train_cluster_losses, "Train")

            test_distances = torch.zeros((len(self.testset), self.n_clusters))
            test_loader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=True)
            test_cluster_losses, test_ae_losses = [], []
            self.deep_cluster_net.eval()
            with torch.no_grad():
                if self.polar_mapping_enabled and epoch > 0:
                    test_polarmapper.set_center(test_embeds.to(self.device))
                for test_indices, test_batch, _ in test_loader:
                    test_batch = test_batch.to(self.device)

                    # Eval one mini-batch
                    test_kmeans_distance, test_batch_distances, test_reconstruction, test_embedding_, \
                                          test_polar_embed_ = self.deep_cluster_net.forward(test_batch, test_polarmapper)
                    test_ae_loss = self.deep_cluster_net.autoencoder.get_autoencoder_loss(test_reconstruction, test_batch)\
                        .item()
                    test_cluster_loss = test_kmeans_distance.sum(dim=1).mean().item()

                    # Save metrics to cpu memory
                    test_ae_losses.append(test_ae_loss)
                    test_cluster_losses.append(test_cluster_loss)
                    test_distances[test_indices, :] = test_batch_distances.detach().cpu().float()
                    test_embeds[test_indices, :] = test_embedding_.detach().cpu().float()
                    if self.polar_mapping_enabled:
                        test_polar_embeds[test_indices, :] = test_polar_embed_.detach().cpu().float()

                del test_batch, test_kmeans_distance, test_batch_distances, test_reconstruction, test_ae_loss, \
                    test_cluster_loss

            self._print_finetune_metrics(test_distances, test_ae_losses, test_cluster_losses, "Test")
            self._log("")

        # Visualize embedding space
        if self.is_writing_to_disc:
            test_cluster_labels = self.infer_cluster_label(test_distances)
            if self.polar_mapping_enabled:
                self._cluster_plot(test_embeds, test_cluster_labels, "finetune", show_centers=False)
                self._cluster_plot(test_polar_embeds, test_cluster_labels, "finetune_polar")
            else:
                self._cluster_plot(test_embeds, test_cluster_labels, "finetune")

    def _cluster_plot(self, test_embeds, test_cluster_labels, phase,  show_centers=True):
        if phase == "pretrain" or "pretrain_polar":
            cluster_centers = torch.from_numpy(self.kmeans_model.cluster_centers_)
        else:
            cluster_centers = self.deep_cluster_net.cluster_reps.detach().cpu().float()
        centers_and_embeds = torch.cat((test_embeds, cluster_centers))
        if test_embeds.shape[1] > 2:
            mapped_embeds_and_centers = PCA(n_components=2).fit_transform(centers_and_embeds)
        elif test_embeds.shape[1] == 2:
            mapped_embeds_and_centers = centers_and_embeds.clone()
        else:
            zeros = torch.zeros_like(centers_and_embeds)
            mapped_embeds_and_centers = torch.cat((centers_and_embeds, zeros), dim=1)
        mapped_embeds = mapped_embeds_and_centers[:len(test_embeds)]
        mapped_centers = mapped_embeds_and_centers[len(test_embeds):]
        label_map = get_clusterlabel_to_groundtruth_map(self.testset.target, test_cluster_labels)
        centers_ground_truths = [label_map[i] for i in range(len(cluster_centers))]
        color_map = get_color_map(self.n_clusters)
        darkened_color_map = get_color_map(self.n_clusters, is_darker=True)
        embeds_colors = [darkened_color_map[t.item()] for t in self.testset.target]
        centers_colors = [color_map[gt] for gt in centers_ground_truths]
        fig, ax = plt.subplots()
        for i in range(len(self.testset)):
            if self.dataset_name == "MNIST":
                ax.plot(mapped_embeds[i, 0], mapped_embeds[i, 1], ".", color=embeds_colors[i],
                        label=f"{self.testset.target[i]}", alpha=0.3)
            else:
                ax.plot(mapped_embeds[i, 0], mapped_embeds[i, 1], ".", color=embeds_colors[i],
                        label=f"{self.inverse_disease_mapping[self.testset.target[i].item()]}", alpha=0.3)
        if show_centers:
            for i in range(len(cluster_centers)):
                ax.plot(mapped_centers[i, 0], mapped_centers[i, 1], self._get_trimarker(i), color=centers_colors[i],
                        zorder=2)
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        self._legend_without_duplicate_labels(ax, loc="upper center", bbox_to_anchor=(0.5, -0.05), fancybox=True)
        fig.savefig(self.directory / f"torch_centers_run{self.run}_{phase}.jpg")

    @staticmethod
    def _legend_without_duplicate_labels(ax, **kwargs):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        for h, l in unique:
            h.set_alpha(1)
        ax.legend(*zip(*unique), **kwargs)

    @staticmethod
    def _get_trimarker(index):
        markers = ["v", "^", "<", ">"]
        marker_id = (index % 4)
        return markers[marker_id]

    def _visualize_autoencoder_image(self):
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

    def _visualize_autoencoder_matrix(self):
        index = np.random.choice(len(self.testset), size=1)
        rand_input = self.testset[index][1].to(self.device)
        with torch.no_grad():
            reconstruction = self.autoencoder.forward(rand_input)[1].cpu().detach().numpy()
        rand_input = rand_input.cpu().detach().numpy()
        if self.ae_objective == "PREDICTION":
            prediction_size = reconstruction.shape[-1]
            rand_input = rand_input[:, :, :, prediction_size:]
        fig, axes = plt.subplots(3, 4)
        x = torch.arange(reconstruction.shape[-1])
        lead_names = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'vx', 'vy', 'vz']
        for lead in range(self.img_height):
            plot_row = lead // 4
            plot_col = lead % 4
            ax = axes[plot_row, plot_col]
            ax.plot(x, rand_input[0, 0, lead, :], color="blue")
            ax.plot(x, reconstruction[0, 0, lead, :], color="red")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"lead {lead_names[lead]}")
        fig.savefig(self.directory / f"matrix_{type(self.autoencoder_setup)}_{self.run}.jpg")

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

    def save_architecture(self):
        with self.directory.joinpath("autoencoder.txt").open("w+") as f:
            f.write(str(self.autoencoder_setup))
