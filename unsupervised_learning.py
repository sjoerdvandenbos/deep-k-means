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
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from susi import SOMClustering
from torch.utils.data import DataLoader
from torch_two_sample.statistics_diff import MMDStatistic

from log_metrics import write_visuals_and_summary
from utils import cluster_acc, map_clusterlabels_to_groundtruth, get_color_map, get_clusterlabel_to_groundtruth_map
from ptb_img_utils import reconstruct_image
from modules.misc import DeepKMeans, PolarMapper
from learning_super import Learner


class UnsupervisedLearner(Learner):

    def __init__(self, args, specs, autoencoder_setup):
        super().__init__(args, specs)
        self.autoencoder_setup = autoencoder_setup
        self.polar_mapping_enabled = args.polar_mapping
        self._log(f"polar_mapping_enabled={self.polar_mapping_enabled}")
        self._log(f"autoencder setup:\n{autoencoder_setup}")
        self.deep_cluster_net = None
        self.optimizer = None

    def run_repeated_learning(self):
        for _ in range(self.n_runs):
            self._log(f"Run {self.run}")
            self.autoencoder_setup.set_new_autoencoder()
            self.optimizer = torch.optim.Adam(self.autoencoder_setup.parameters(), self.lr)
            if self.is_writing_to_disc and self.run == 0:
                print(self.autoencoder_setup.autoencoder)
                self.save_architecture()
            if self.seeded:
                torch.manual_seed(self.seeds[self.run])
                np.random.seed(self.seeds[self.run])
            self.pretrain()
            if self.n_finetuning_epochs > 0:
                self.finetune()
            if self.autoencoder_setup.name == "VARIATIONAL_AUTOENCODER":
                # Generate 3000 ECGs
                with torch.no_grad():
                    generated_samples = self.autoencoder_setup.generate(3000).detach().cpu()
                # Calc MMD between generated and test set
                rand_sample_index = np.random.choice(3000)
                rand_vec = generated_samples[rand_sample_index, :]
                self._visualize_single_vector(rand_vec)

            self.run += 1
        duration = datetime.now() - self.start_time
        self._log(f"Learning duration: {duration}")
        write_visuals_and_summary(self.directory)

    def pretrain(self):
        for epoch in range(self.n_pretrain_epochs):
            self._log(f"Pretraining step: epoch {epoch}")
            train_loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
            train_polarmapper = PolarMapper(self.embedding_size, self.device)
            train_embeds = torch.zeros((len(self.trainset), self.embedding_size), dtype=torch.float, device=self.device)
            train_polar_embeds = torch.zeros_like(train_embeds, dtype=torch.float, device=self.device)
            ae_losses = []
            self.autoencoder_setup.train()
            for train_indices, train_batch, _ in train_loader:
                train_batch = train_batch.to(self.device)
                self.pretrain_minibatch(train_indices, train_batch, train_embeds, ae_losses, train_polar_embeds,
                                        train_polarmapper)

            test_loader = DataLoader(self.testset, batch_size=self.test_size, shuffle=True, pin_memory=True)
            test_polarmapper = PolarMapper(self.embedding_size, self.device)
            test_embeds = torch.zeros((len(self.testset), self.embedding_size), dtype=torch.float, device=self.device)
            test_polar_embeds = torch.zeros_like(test_embeds, device=self.device)
            test_ae_losses = []
            self.autoencoder_setup.eval()
            with torch.no_grad():
                for test_indices, test_batch, _ in test_loader:
                    test_batch = test_batch.to(self.device)
                    self.pretrain_minibatch(test_indices, test_batch, test_embeds, test_ae_losses,
                                            test_polar_embeds, test_polarmapper)

            if self.polar_mapping_enabled:
                self.kmeans_model = KMeans(n_clusters=self.n_clusters, tol=1e-6)
                self.kmeans_model.fit(train_polar_embeds.detach().double().cpu().numpy())
                train_cluster_labels = self.kmeans_model.labels_.astype(int)
                test_cluster_labels = torch.from_numpy(self.kmeans_model.predict(test_polar_embeds.detach().double().cpu()))
            else:
                self.kmeans_model = KMeans(n_clusters=self.n_clusters, tol=1e-6)
                self.kmeans_model.fit(train_embeds.detach().double().cpu().numpy())
                train_cluster_labels = self.kmeans_model.labels_.astype(int)
                # train_bmus = self.kmeans_model.get_bmus(train_embeds.detach().double().cpu().numpy())
                # train_cluster_labels = np.array([5*r + c for r, c in train_bmus])
                test_cluster_labels = torch.from_numpy(self.kmeans_model.predict(test_embeds.detach().double().cpu()))
                # test_bmus = self.kmeans_model.get_bmus(test_embeds.detach().double().cpu().numpy())
                # test_cluster_labels = np.array([5*r + c for r, c in test_bmus])


            self._log(f"Train auto encoder loss: {sum(ae_losses) / self.n_batches}")
            self._print_cluster_metrics(self.trainset.target, train_cluster_labels, "Train")
            # self._print_mmd("Train")
            self._log(f"Test auto encoder loss: {sum(test_ae_losses) / self.n_test_batches}")
            self._print_cluster_metrics(self.testset.target, test_cluster_labels, "Test")
            # self._print_mmd("Test")
            self._log("")

        if self.is_writing_to_disc:
            np.save(self.directory / f"train_embeds{self.run}.npy", train_embeds.cpu().numpy())
            np.save(self.directory / f"test_embeds{self.run}.npy", test_embeds.cpu().numpy())
            if self.polar_mapping_enabled:
                np.save(self.directory / "test_polar_embeds.npy", test_polar_embeds.cpu().numpy())
            self.draw_visualizations(
                "pretrain",
                test_embeds.detach().cpu().double(),
                test_polar_embeds.detach().cpu().double(),
                test_cluster_labels,
            )

    def pretrain_minibatch(self, indices, batch, embeds, ae_losses, mapped_embeds, mapper):
        self.autoencoder_setup.zero_grad(set_to_none=True)
        if self.autoencoder_setup.name == "VARIATIONAL_AUTOENCODER":
            embed_, reconstruction, means, log_var = self.autoencoder_setup.forward(batch)
        else:
            embed_, reconstruction = self.autoencoder_setup.forward(batch)
        embeds[indices, :] = embed_.detach()
        with torch.no_grad():
            if self.polar_mapping_enabled:
                mapper.update(embeds)
                mapped_embeds[indices, :] = mapper(embed_).detach()
        if self.autoencoder_setup.name == "VARIATIONAL_AUTOENCODER":
            train_ae_loss_ = self.autoencoder_setup.get_autoencoder_loss(reconstruction, batch, means, log_var)
        else:
            train_ae_loss_ = self.autoencoder_setup.get_autoencoder_loss(reconstruction, batch)
        if self.autoencoder_setup.training:
            train_ae_loss_.backward()
            torch.nn.utils.clip_grad_value_(self.autoencoder_setup.parameters(), 10.0)
            self.optimizer.step()
        ae_losses.append(train_ae_loss_.item())

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
            train_polarmapper = PolarMapper(self.embedding_size, self.device)
            test_polarmapper = PolarMapper(self.embedding_size, self.device)
        else:
            train_polarmapper = None
            test_polarmapper = None
        for epoch in range(self.n_finetuning_epochs):
            # if self.polar_mapping_enabled and epoch > 0:
            #     train_polarmapper.set_center(train_embeds.to(self.device))
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

        if self.is_writing_to_disc:
            test_cluster_labels = self.infer_cluster_label(test_distances)
            self.draw_visualizations("finetune", test_embeds, test_polar_embeds, test_cluster_labels)

    def draw_visualizations(self, phase, embeds, mapped_embeds, test_cluster_labels):
        if self.data_format == "IMAGE":
            self._visualize_autoencoder_image()
        elif self.data_format == "MATRIX":
            self._visualize_autoencoder_matrix()
        elif self.data_format == "VECTOR":
            self._visualize_autoencoder_vector()
        elif self.data_format == "IMAGE_FIELDS":
            self._visualize_image_fields()
        if self.polar_mapping_enabled:
            self._cluster_plot_diseases(embeds, test_cluster_labels, phase, show_centers=False)
            self._cluster_plot_diseases(mapped_embeds, test_cluster_labels, f"{phase}_polar")
        else:
            self._cluster_plot_diseases(embeds, test_cluster_labels, "pretrain", show_centers=True)

    def _cluster_plot_diseases(self, test_embeds, test_cluster_labels, phase, show_centers=False):
        mapped_embeds_and_centers = self._apply_dimensionality_reduction(test_embeds, phase, show_centers)
        mapped_embeds = mapped_embeds_and_centers[:len(test_embeds)]
        label_map = get_clusterlabel_to_groundtruth_map(self.testset.target, test_cluster_labels, self.n_clusters)
        darkened_color_map = get_color_map(self.n_clusters, is_darker=True)
        embeds_colors = [darkened_color_map[t.item()] for t in self.testset.target]
        fig, ax = plt.subplots()
        for i in range(len(self.testset)):
            ax.plot(mapped_embeds[i, 0], mapped_embeds[i, 1], ".", color=embeds_colors[i],
                    label=f"{self.inverse_disease_mapping[self.testset.target[i].item()]}", alpha=0.3)
        if show_centers:
            mapped_centers = mapped_embeds_and_centers[len(test_embeds):]
            centers_ground_truths = torch.tensor([label_map[i] for i in range(self.n_clusters)])
            color_map = get_color_map(self.n_clusters)
            centers_colors = [color_map[gt.item()] for gt in centers_ground_truths]
            for i in range(self.n_clusters):
                ax.plot(mapped_centers[i, 0], mapped_centers[i, 1], self._get_trimarker(i), color=centers_colors[i],
                        zorder=2)
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        self._legend_without_duplicate_labels(ax, bbox_to_anchor=(0.75, 1), loc="upper left", fancybox=True)
        fig.savefig(self.directory / f"torch_centers_run{self.run}_{phase}.jpg")
        fig.clear()

    def _cluster_plot_others(self, embeds, targets, name):
        mapped_embeds = self._apply_dimensionality_reduction(embeds, "custom")
        inverse_class_mapping = dict(enumerate(targets.unique()))
        class_mapping = {v: k for k, v in inverse_class_mapping.items()}
        darkened_color_map = get_color_map(len(class_mapping), is_darker=True)
        embeds_colors = [darkened_color_map[class_mapping[t]] for t in targets]
        fig, ax = plt.subplots()
        for i in range(embeds.shape[0]):
            ax.plot(mapped_embeds[i, 0], mapped_embeds[i, 1], ".", color=embeds_colors[i],
                    label=f"{targets[i]}", alpha=0.3)
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        self._legend_without_duplicate_labels(ax, bbox_to_anchor=(0.75, 1), loc="upper left", fancybox=True)
        fig.savefig(self.directory / f"torch_centers_run{self.run}_{name}.jpg")
        fig.clear()

    def _apply_dimensionality_reduction(self, embeds, phase, include_centers=True):
        if phase == "pretrain" or phase == "pretrain_polar":
            cluster_centers = torch.from_numpy(self.kmeans_model.cluster_centers_)
        else:
            cluster_centers = self.deep_cluster_net.cluster_reps.detach().cpu().float()
        if include_centers:
            vecs = torch.cat((embeds, cluster_centers))
        else:
            vecs = embeds.clone()
        # vecs = embeds.clone()
        if embeds.shape[1] > 2:
            return PCA(n_components=2).fit_transform(vecs)
        elif embeds.shape[1] == 2:
            return vecs.clone()
        else:
            zeros = torch.zeros_like(vecs)
            return torch.cat((vecs, zeros), dim=1)

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
        rand_input, reconstruction = self._get_random_input_and_reconstruction()
        input_img = reconstruct_image(rand_input.numpy(), img_shape)
        output_img = reconstruct_image(reconstruction.numpy(), img_shape, transform=True)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(input_img)
        ax1.set_title("input")
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax2.imshow(output_img)
        ax2.set_title("reconstruction")
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        fig.savefig(self.directory / f"Image_reconstructions_run_{self.run}.jpg")
        fig.clear()

    def _visualize_image_fields(self):
        img_shape = [self.img_height, self.img_width]
        rand_input = self._get_random_input()
        reconstruction = self._get_reconstruction(rand_input).numpy()
        rand_input = rand_input.cpu().numpy()
        input_channels = np.split(rand_input, 3, axis=1)
        reconstruction_channels = np.split(reconstruction, 3, axis=1)
        fig, axes = plt.subplots(3, 2)
        for i, (iput, reconstruction) in enumerate(zip(input_channels, reconstruction_channels)):
            ax_in = axes[i, 0]
            ax_out = axes[i, 1]
            input_img = reconstruct_image(iput, img_shape)
            output_img = reconstruct_image(reconstruction, img_shape, transform=True)
            ax_in.imshow(input_img)
            ax_out.imshow(output_img)
            ax_in.set_title("Input")
            ax_in.get_xaxis().set_visible(False)
            ax_in.get_yaxis().set_visible(False)
            ax_out.set_title("Output")
            ax_out.get_xaxis().set_visible(False)
            ax_out.get_yaxis().set_visible(False)
        fig.savefig(self.directory / f"Image_fields_reconstructions_run_{self.run}.jpg")
        fig.clear()

    def _visualize_autoencoder_vector(self):
        rand_input, reconstruction = self._get_random_input_and_reconstruction()
        fig, ax = plt.subplots(1, 1)
        ax.plot(torch.arange(rand_input.shape[-1]), rand_input[0, 0, 0, :], "b", label="input")
        ax.plot(torch.arange(reconstruction.shape[-1]), reconstruction[0, 0, 0, :], "r", label="reconstruction")
        ax.set_title("vector reconstruction")
        ax.legend()
        fig.savefig(self.directory / f"vector_{self.autoencoder_setup!r}_{self.run}.jpg")
        fig.clear()

    def _visualize_autoencoder_matrix(self):
        rand_input, reconstruction = self._get_random_input_and_reconstruction()
        x1 = torch.arange(reconstruction.shape[-1])
        input_format = self.autoencoder_setup.get_input_format()
        if self.autoencoder_setup.objective == "PREDICTION" and input_format == "RNN":
            prediction_size = reconstruction.shape[3]
            rand_input = rand_input[:, :, :, -prediction_size:]
            x1 = torch.arange(rand_input.shape[3])
        elif self.autoencoder_setup.objective == "RECONSTRUCTION" and input_format == "RNN":
            start_indices = rand_input.shape[3] - reconstruction.shape[3]
            reconstruction = torch.flip(reconstruction, (3,))
            rand_input = rand_input[:, :, :, start_indices:]
            x1 = torch.arange(rand_input.shape[3])
        elif self.autoencoder_setup.objective == "HYBRID" and input_format == "RNN":
            reconstruction_truth, prediction_truth = self.autoencoder_setup.get_hybrid_target(rand_input)
            rand_input = torch.cat((reconstruction_truth, prediction_truth), dim=3)
            x1 = torch.arange(rand_input.shape[3])
        n_cols = self.img_height // 3
        fig, axes = plt.subplots(3, n_cols)
        x2 = torch.arange(reconstruction.shape[-1])
        lead_names = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'vx', 'vy', 'vz']
        for lead in range(self.img_height):
            plot_row = lead // n_cols
            plot_col = lead % n_cols
            ax = axes[plot_row, plot_col]
            ax.plot(x1, rand_input[0, 0, lead, :], "-b", alpha=0.5)
            ax.plot(x2, reconstruction[0, 0, lead, :], ":r", alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"lead {lead_names[lead]}")
        fig.savefig(self.directory / f"matrix{self.autoencoder_setup!r}_{self.run}.jpg")
        fig.clear()

    def _visualize_single_vector(self, vec):
        fig, ax = plt.subplots(1, 1)
        ax.plot(torch.arange(vec.shape[0]), vec, "b")
        ax.set_title("Generated Vector")
        ax.legend()
        fig.savefig(self.directory / f"generated_vector_{self.run}.jpg")
        fig.clear()

    def _get_random_input_and_reconstruction(self):
        rand_input = self._get_random_input()
        reconstruction = self._get_reconstruction(rand_input)
        rand_input = rand_input.cpu().detach()
        return rand_input, reconstruction

    def _get_random_input(self):
        index = np.random.choice(len(self.testset), size=1)
        rand_input = self.testset[index][1].to(self.device)
        return rand_input

    def _get_reconstruction(self, iput):
        with torch.no_grad():
            reconstruction = self.autoencoder_setup.forward(iput)[1].cpu().detach()
        return reconstruction

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
        predictions = map_clusterlabels_to_groundtruth(ground_truths, cluster_labels, self.n_clusters)
        acc = cluster_acc(ground_truths, predictions)
        self._log(f"{phase} ACC: {acc}")
        ari = adjusted_rand_score(ground_truths, predictions)
        self._log(f"{phase} ARI: {ari}")
        ids = self.test_ids if phase == "Test" else self.train_ids
        nmi = normalized_mutual_info_score(cluster_labels, ids)
        self._log(f"{phase} NMI: {nmi}")

    def _print_mmd(self, phase):
        # Generate 3000 ECGs
        with torch.no_grad():
            generated_samples = self.autoencoder_setup.generate(3000).detach().cpu()
        # Calc MMD between generated and test set
        if phase == "Test":
            real_samples = self.testset.data.squeeze()
        else:
            real_samples = self.trainset.data.squeeze()
        # Calculate hyperparam alpha as the median pairwise distance between the joint distributions
        distance_matrix = pairwise_distances(generated_samples, real_samples)
        median = np.median(distance_matrix)
        mmd_metric = MMDStatistic(3000, real_samples.shape[0])(generated_samples, real_samples, [median.item()])
        self._log(f"{phase} MMD: {mmd_metric}")

    @staticmethod
    def infer_cluster_label(distances):
        return torch.argmin(distances, dim=1)

    def save_architecture(self):
        with self.directory.joinpath("autoencoder.txt").open("w+") as f:
            f.write(str(self.autoencoder_setup))

    @staticmethod
    def reduce_dimensionality_tsne(embeds):
        numpy_embeds = embeds.detach().double().cpu().numpy()
        return TSNE(n_components=2).fit(numpy_embeds)
