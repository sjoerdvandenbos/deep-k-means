from torch import nn
import torch


class DeepKMeans(nn.Module):
    def __init__(self, autoencoder, cluster_reps):
        super().__init__()
        self.cluster_reps = torch.nn.Parameter(cluster_reps)                                       # Learnable param
        self.autoencoder = autoencoder

        self.n_clusters = torch.nn.Parameter(torch.tensor(cluster_reps.shape[0]), requires_grad=False)
        self.embedding_size = torch.nn.Parameter(torch.tensor(cluster_reps.shape[1]), requires_grad=False)

    def forward(self, x):
        """
        Tensors are expanded to shape [batch_size, embedding_size, embedding_size] and later to shape [batch_size,
        embedding_size] in order to facilitate element-wise operations.
        """
        batch_size: int = x.shape[0]
        embeddings, reconstruction = self.autoencoder.forward(x)
        distances = self._get_distances(embeddings, batch_size)
        # distances.shape = [batch_size, n_clusters]
        min_dist = distances.min(dim=1, keepdim=True)[0].expand(-1, self.n_clusters)
        # min_dist.shape = [batch_size, n_clusters]
        alpha: float = 1000.
        exponentials = self._compute_shifted_exps(alpha, distances, min_dist)
        # exponentials.shape = [batch_size, n_clusters]
        sum_exponentials = exponentials.sum(dim=1, keepdim=True).expand(-1, self.n_clusters)
        # sum_exponentials.shape = [batch_size, n_clusters]
        weighted_dists = self._compute_weighted_dists(distances, exponentials, sum_exponentials)
        # weighted_dists.shape = [batch_size, n_clusters]
        return weighted_dists, distances, reconstruction, embeddings

    def _get_distances(self, embeddings: torch.Tensor, batch_size: int):
        """ Returns distances between embeddings and cluster reps. """
        repeated_cluster_reps = self.cluster_reps.unsqueeze(0).expand(batch_size, -1, -1)
        # repeated_cluster_reps.shape = [batch_size, n_clusters, embedding_size]
        repeated_embedding = embeddings.unsqueeze(1).expand(-1, self.n_clusters, -1)
        # repeated_embedding.shape = [batch_size, n_clusters, embedding_size]
        res = self.cluster_distance(repeated_embedding, repeated_cluster_reps)
        return res.squeeze()

    @staticmethod
    def cluster_distance(x: torch.Tensor, y: torch.Tensor):
        return torch.square(x - y).sum(dim=2, keepdim=True)

    @staticmethod
    def _compute_shifted_exps(alpha: float, distances: torch.Tensor, min_dist: torch.Tensor):
        """ Compute exponentials shifted with min_dist to avoid underflow (0/0) issues in softmaxes. """
        return torch.exp(-alpha * (distances - min_dist))

    @staticmethod
    def _compute_weighted_dists(distances: torch.Tensor, exponentials: torch.Tensor, sum_exponentials:
                                torch.Tensor):
        """
        Returns softmaxes and the embedding/representative distances weighted by softmax
        calculated as distances * softmaxes.
        """
        return distances * (exponentials / sum_exponentials)

