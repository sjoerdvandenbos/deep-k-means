from torch import nn
import torch


class FCAutoencoder(nn.Module):
    def __init__(self, dimensions, height, width):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
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
            ReshapeModule(-1, 1, height, width)
        )

    def forward(self, input):
        embedding = self.encoder(input)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction


class ConvoAutoencoder(nn.Module):
    def __init__(self, height, width, embed_size):
        super().__init__()
        padding1 = 2
        stride1 = 2
        kernel_size1 = 5
        output_height_padding_1 = self._get_output_padding(height)
        output_width_padding_1 = self._get_output_padding(width)
        post_conv_height_1 = self._get_post_conv_length(height, padding1, kernel_size1, stride1)
        post_conv_width_1 = self._get_post_conv_length(width, padding1, kernel_size1, stride1)

        padding2 = 2
        stride2 = 2
        kernel_size2 = 5
        output_height_padding_2 = self._get_output_padding(post_conv_height_1)
        output_width_padding_2 = self._get_output_padding(post_conv_width_1)
        post_conv_height_2 = self._get_post_conv_length(post_conv_height_1, padding2, kernel_size2, stride2)
        post_conv_width_2 = self._get_post_conv_length(post_conv_width_1, padding2, kernel_size2, stride2)

        padding3 = 0
        stride3 = 2
        kernel_size3 = 3
        output_height_padding_3 = self._get_output_padding(post_conv_height_2)
        output_width_padding_3 = self._get_output_padding(post_conv_width_2)
        post_conv_height_3 = self._get_post_conv_length(post_conv_height_2, padding3, kernel_size3, stride3)
        post_conv_width_3 = self._get_post_conv_length(post_conv_width_2, padding3, kernel_size3, stride3)

        linear_size = post_conv_height_3 * post_conv_width_3 * 128

        self.encoder = nn.ModuleList([
            nn.Conv2d(1, 32, kernel_size=(kernel_size1, kernel_size1), stride=(stride1, stride1),
                      padding=(padding1, padding1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(kernel_size2, kernel_size2), stride=(stride2, stride2),
                      padding=(padding2, padding2)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(kernel_size3, kernel_size3), stride=(stride3, stride3),
                      padding=(padding3, padding3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(linear_size, embed_size),
        ])

        self.decoder = nn.ModuleList([
            nn.Linear(embed_size, linear_size),
            ReshapeModule(-1, 128, post_conv_height_3, post_conv_width_3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(kernel_size3, kernel_size3), stride=(stride3, stride3),
                               padding=(padding3, padding3),
                               output_padding=(output_height_padding_3, output_width_padding_3)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(kernel_size2, kernel_size2), stride=(stride2, stride2),
                               padding=(padding2, padding2),
                               output_padding=(output_height_padding_2, output_width_padding_2)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(kernel_size1, kernel_size1), stride=(stride1, stride1),
                               padding=(padding1, padding1),
                               output_padding=(output_height_padding_1, output_width_padding_1)),
        ])

    def forward(self, x):
        for m in self.encoder:
            x = m(x)
        embedding = x
        for m in self.decoder:
            x = m(x)
        return embedding, x

    @staticmethod
    def _get_post_conv_length(before, padding, kernel, stride):
        after_conv = (before + 2 * padding - kernel) // stride + 1
        return after_conv

    @staticmethod
    def _get_output_padding(before):
        if before % 2 == 0:
            return 1
        else:
            return 0


class DeepKMeans(nn.Module):
    def __init__(self, autoencoder, cluster_reps):
        super().__init__()
        cluster_reps.requires_grad = True
        self.cluster_reps = torch.nn.Parameter(cluster_reps)
        self.autoencoder = autoencoder

    def _apply(self, fn):
        super()._apply(fn)
        fn(self.cluster_reps)

    def forward(self, x):
        """
        Tensors are expanded to shape [batch_size, embedding_size, embedding_size] and later to shape [batch_size,
        embedding_size] in order to facilitate element-wise operations.
        """
        embedding_size = self.cluster_reps.shape[0]
        batch_size = x.shape[0]
        embedding, reconstruction = self.autoencoder.forward(x)
        distances = self._get_distances(embedding, batch_size)
        # distances.shape = [batch_size, embedding_size]
        min_dist = distances.min(dim=1, keepdim=True)[0].expand(-1, embedding_size)
        # print(f"distances: {distances[0, :]}")
        # print("\n\n")
        # print(f"min_dist: {min_dist[0, :]}")
        # min_dist.shape = [batch_size, embedding_size]
        alpha = 1000.
        exponentials = self._compute_shifted_exps(alpha, distances, min_dist)
        # print("\n\n")
        # print(f"exponentials: {exponentials[0, :]}")
        # exponentials.shape = [batch_size, embedding_size]
        sum_exponentials = exponentials.sum(dim=1, keepdim=True).expand(-1, embedding_size)
        # sum_exponentials.shape = [batch_size, embedding_size]
        # print("\n\n")
        # print(f"sum exponentials: {sum_exponentials[0, :]}")
        weighted_dists = self._compute_weighted_dists(distances, exponentials, sum_exponentials)
        # print("\n\n")
        # print(f"weighted_dists: {weighted_dists[0, :]}")
        # exit()
        # weighted_dists.shape = [batch_size, embedding_size]
        return weighted_dists, distances, reconstruction

    def _get_distances(self, embedding, batch_size):
        """ Returns distances between embeddings and cluster reps. """
        embedding_size = self.cluster_reps.shape[0]
        repeated_cluster_reps = self.cluster_reps.view(1, embedding_size, embedding_size) \
                                                   .expand(batch_size, -1, -1)
        # print(f"repeated_cluster_reps: {repeated_cluster_reps[0, :, :]}")
        # print("\n\n")
        repeated_embedding = embedding.unsqueeze(1).expand(-1, embedding_size, -1)
        # print(f"repeated_embeddings: {repeated_embedding[0 ,:, :]}")
        res = self.cluster_distance(repeated_embedding, repeated_cluster_reps)
        # print("\n\n")
        # print(f'result: {res[0, :]}')
        # exit()
        return res.squeeze()

    @staticmethod
    def cluster_distance(x: torch.Tensor, y: torch.Tensor):
        return torch.square(x - y).sum(dim=2, keepdim=True)

    @staticmethod
    def _compute_shifted_exps(alpha: float, distances: torch.Tensor, min_dist: torch.Tensor):
        """ Compute exponentials shifted with min_dist to avoid underflow (0/0) issues in softmaxes. """
        return torch.exp(-alpha * (distances - min_dist))

    @staticmethod
    def _compute_weighted_dists(distances: torch.Tensor, exponentials: torch.Tensor, sum_exponentials: torch.Tensor):
        """
        Returns softmaxes and the embedding/representative distances weighted by softmax
        calculated as distances * softmaxes.
        """
        return distances * (exponentials / sum_exponentials)


class ReshapeModule(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.reshape(*self.args)


class OrthogonalLinearModule(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.has_bias = bias
        self.n_groups = self._get_number_groups()
        self.group_size = self.out_features // self.n_groups

    def forward(self, x):
        weights = self.linear.weight.view((self.n_groups, self.group_size, self.in_features))
        w_center = weights - weights.mean(dim=-1, keepdim=True)
        del weights
        cov_mat = w_center.bmm(torch.transpose(w_center, dim0=1, dim1=2))
        mappings = self._get_orth_mapping(cov_mat)
        del cov_mat
        new_weights = mappings.bmm(w_center).view((self.out_features, self.in_features))
        del w_center
        return x.mm(new_weights.T) + self.linear.bias if self.has_bias else 0

    @staticmethod
    def _get_orth_mapping(cov_mat):
        eigen_vecs, eigen_vals, _ = torch.linalg.svd(cov_mat)
        diagonal = eigen_vals.rsqrt().diag_embed()
        return eigen_vecs.bmm(diagonal).bmm(torch.transpose(eigen_vecs, dim0=1, dim1=2))

    def _get_number_groups(self):
        group_size = 2
        while self.out_features % group_size != 0:
            group_size += 1
        return self.out_features // group_size


class OLMAutoencoder(nn.Module):
    def __init__(self, dimensions, height, width):
        super(OLMAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            OrthogonalLinearModule(dimensions[-1], dimensions[0]),
            nn.ReLU(),
            OrthogonalLinearModule(dimensions[0], dimensions[1]),
            nn.ReLU(),
            OrthogonalLinearModule(dimensions[1], dimensions[2]),
            nn.ReLU(),
            OrthogonalLinearModule(dimensions[2], dimensions[3]),
        )
        self.decoder = nn.Sequential(
            OrthogonalLinearModule(dimensions[3], dimensions[4]),
            nn.ReLU(),
            OrthogonalLinearModule(dimensions[4], dimensions[5]),
            nn.ReLU(),
            OrthogonalLinearModule(dimensions[5], dimensions[6]),
            nn.ReLU(),
            OrthogonalLinearModule(dimensions[6], dimensions[7]),
            ReshapeModule(-1, 1, height, width)
        )

    def forward(self, input):
        embedding = self.encoder(input)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction

