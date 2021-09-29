from torch import nn
import torch

from ptb_img_utils import kmeans_distance


class FCAutoencoder(nn.Module):
    def __init__(self, dimensions):
        super(FCAutoencoder, self).__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(dimensions[-1], dimensions[0]),
            nn.ReLU(),
            nn.Linear(dimensions[0], dimensions[1]),
            nn.ReLU(),
            nn.Linear(dimensions[1], dimensions[2]),
            nn.ReLU(),
            nn.Linear(dimensions[2], dimensions[3]),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dimensions[3], dimensions[4]),
            nn.ReLU(),
            nn.Linear(dimensions[4], dimensions[5]),
            nn.ReLU(),
            nn.Linear(dimensions[5], dimensions[6]),
            nn.ReLU(),
            nn.Linear(dimensions[6], dimensions[7]),
        )

    def forward(self, input):
        embedding = self.encoder(input)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction


class DeepKMeans(nn.Module):
    def __init__(self, autoencoder, cluster_reps):
        super(DeepKMeans, self).__init__()
        self.autoencoder = autoencoder
        self.cluster_reps = cluster_reps

    def forward(self, input):
        embedding, reconstruction = self.autoencoder.forward(input)
        embedding_size = self.cluster_reps.shape[0]
        list_dist = []
        for i in range(embedding_size):
            dist = kmeans_distance(embedding, torch.reshape(self.cluster_reps[i, :], (1, embedding_size)))
            list_dist.append(dist)
        stack_dist = torch.stack(list_dist)
        min_dist, _ = stack_dist.min(dim=0)

        ## Third, compute exponentials shifted with min_dist to avoid underflow (0/0) issues in softmaxes
        alpha = 1000
        list_exp = []
        for i in range(embedding_size):
            exp = torch.exp(-alpha * (stack_dist[i] - min_dist))
            list_exp.append(exp)
        stack_exp = torch.stack(list_exp)
        sum_exponentials = torch.sum(stack_exp, dim=0)

        ## Fourth, compute softmaxes and the embedding/representative distances weighted by softmax
        list_softmax = []
        list_weighted_dist = []
        for j in range(embedding_size):
            softmax = stack_exp[j] / sum_exponentials
            weighted_dist = stack_dist[j] * softmax
            list_softmax.append(softmax)
            list_weighted_dist.append(weighted_dist)
        return torch.stack(list_weighted_dist), reconstruction
