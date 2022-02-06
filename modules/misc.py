from math import pi as PI, ceil

from torch import nn
import torch


class DeepKMeans(nn.Module):
    def __init__(self, autoencoder, cluster_reps):
        super().__init__()
        self.cluster_reps = torch.nn.Parameter(cluster_reps)                                       # Learnable param
        self.autoencoder = autoencoder

        self.n_clusters = torch.nn.Parameter(torch.tensor(cluster_reps.shape[0]), requires_grad=False)
        self.embedding_size = torch.nn.Parameter(torch.tensor(cluster_reps.shape[1]), requires_grad=False)

    def forward(self, x, embedding_mapper=None):
        """
        Tensors are expanded to shape [batch_size, embedding_size, embedding_size] and later to shape [batch_size,
        embedding_size] in order to facilitate element-wise operations.
        """
        batch_size: int = x.shape[0]
        embeddings, reconstruction = self.autoencoder.forward(x)
        if embedding_mapper is not None:
            mapped_embeds = embedding_mapper(embeddings)
        else:
            mapped_embeds = embeddings.clone()
        distances = self._get_distances(mapped_embeds, batch_size)
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
        return weighted_dists, distances, reconstruction, embeddings, mapped_embeds

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


class PolarMapper(nn.Module):
    def __init__(self, embedding_size, device):
        super().__init__()
        self.embeddings_center = torch.zeros((1, embedding_size), device=device)
        self.center_is_defined = False

    def forward(self, x):
        batch_size, embeddings_size = x.shape
        assert embeddings_size == 2, "Only implemented for x and y, so 2 dims"
        center = self._get_center(x)
        # print(f"center.shape={center.shape}")
        differences = x - center.expand(batch_size, -1)
        # print(f"differences.shape={differences.shape}")
        radiusses = differences.square().sum(dim=1).sqrt()
        logradiusses = torch.log(radiusses)
        # Normalize radiusses
        # radiusses = radiusses / radiusses.mean()
        # print(f"radiusses: {radiusses.shape}")
        angles = torch.arctan(differences[:, 1] / (differences[:, 0] + 1e-6))
        # print(f"angles: {angles.shape}")
        result = torch.cat((logradiusses.unsqueeze(1), angles.unsqueeze(1)), dim=1)
        # print(f"result: {result.shape}")
        return result

    def _get_center(self, x):
        if self.center_is_defined:
            return self.embeddings_center
        else:
            return x.mean(dim=0, keepdim=True)

    def update(self, embeds):
        # Ensure there are no more feature vectors (rows) left as zero vectors.
        if torch.all(torch.any(embeds.bool(), dim=1)):
            self.embeddings_center = embeds.mean(dim=0, keepdim=True)
            self.center_is_defined = True


# class PolarMapperAlt1(PolarMapper):
#     """ https://arxiv.org/pdf/2007.10588.pdf Page 4. This mapping assumes the center is at x=0;y=0."""
#     def __init__(self, embedding_size, device):
#         super().__init__(embedding_size, device)
#
#     def forward(self, x):
#         batch_size, embeddings_size = x.shape
#         assert embeddings_size == 2, "Only implemented for x and y, so 2 dims"
#         center = self._get_center(x)
#         differences = x - center.expand(batch_size, -1)
#         x = differences[:, 0].clone()
#         y = differences[:, 1].clone()
#
#     def vectorized_mapping(self, x, y):
#         if x > 0 and y >= 0:
#             return torch.arctan(y / x)
#         elif x == 0 and y >= 0:
#             return torch.full_like(x, PI / 2)
#         elif x < 0 and y >= 0:
#             return PI + torch.arctan(y / x)
#         elif x < 0 and y < 0:
#             return PI + torch.arctan(y / x)
#         elif x == 0 and y < 0:
#             return torch.full_like(x, 3 * PI / 2)
#         elif x > 0 and y < 0:
#             return 2*PI + torch.arctan(y / x)
#         # x == 0 and y == 0
#         else:
#             # Hack this return value to prevent undefined value
#             return torch.full_like(x, PI / 2)


class LSTMDecoderEmpty(nn.LSTM):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0,
                 bidirectional=False, proj_size=0):
        super().__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
                         batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, proj_size=proj_size)
        if bidirectional or proj_size != 0:
            raise NotImplementedError

    def forward(self, x, hidden_and_cell=None):
        empty_input = torch.zeros((x.shape[0], x.shape[1] - 1, x.shape[2]), device=x.device)
        new_input = torch.cat((x[:, 0, :].unsqueeze(1), empty_input), dim=1)
        return super().forward(new_input, hidden_and_cell)


class LSTMDecoderMixedTeacherForcing(nn.LSTM):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0,
                 bidirectional=False, proj_size=0, teacher_forcing_probability=0):
        super().__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
                         batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, proj_size=proj_size)
        if bidirectional or proj_size != 0:
            raise NotImplementedError
        self.teacher_forcing_probability = teacher_forcing_probability

    def forward(self, x, hidden_and_cell=None):
        seq_length = x.shape[1] if self.batch_first else x.shape[0]
        outputs = []
        for t in range(seq_length):
            if torch.rand((1,), device=x.device) < self.teacher_forcing_probability or t == 0:
                temp_input = x[:, t, :].unsqueeze(1)
            else:
                temp_input = out
            out, hidden_and_cell = super().forward(temp_input, hidden_and_cell)
            outputs.append(out)
        return torch.cat(outputs, dim=1), hidden_and_cell


class DualLSTMDecoderAdapter(nn.Module):
    def __init__(self, reconstruction_decoder, prediction_decoder):
        super().__init__()
        self.reconstruction_decoder = reconstruction_decoder
        self.prediction_decoder = prediction_decoder
        self.input_size = reconstruction_decoder.input_size
        self.hidden_size = reconstruction_decoder.hidden_size
        self.num_layers = reconstruction_decoder.num_layers

    def forward(self, x, hidden_and_cell=None):
        # last output of the encoder
        last_output = x[:, 0, :].unsqueeze(1)
        # length of the encoder and length of the decoder
        length = x.shape[1] // 2
        # divide input into a reconstruction part and a future part
        reconstruction_part = x[:, :length, :]
        future_part = torch.cat((last_output, x[:, length:-1, :]), dim=1)
        # forward pass of the reconstruction part
        reconstruction, _ = self.reconstruction_decoder(reconstruction_part, hidden_and_cell)
        # forward pass of the prediction part
        prediction, _ = self.prediction_decoder(future_part, hidden_and_cell)
        # reconstruction is flipped back to the initial order
        reconstruction = reconstruction.flip(1)
        output = torch.cat((reconstruction, prediction), dim=1)
        return output, None
