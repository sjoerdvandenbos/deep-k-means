import torch


def autoencoder_loss(x, y):
    return torch.square(x - y).sum(dim=0).mean()


def f1_loss(x, reconstruction, epsilon=1e-8):
    # Iput range: [0, 1]
    # Reconstruction range: unknown
    reconstruction_binary = _map_range_to_01(reconstruction)
    intersection = torch.sum(x * reconstruction_binary, dim=0)
    denominator = torch.sum(x + reconstruction_binary, dim=0)
    f1 = (2. * intersection + epsilon) / (denominator + epsilon)
    return torch.mean(1. - f1)


def jaccard_loss(x, reconstruction, epsilon=1e-8):
    # Iput range: [0, 1]
    # Reconstruction range: unknown
    reconstruction_binary = _map_range_to_01(reconstruction)
    intersection = x * reconstruction_binary
    union = (x + reconstruction_binary) - intersection
    jac = torch.sum((intersection + epsilon) / (union + epsilon), dim=0)
    return torch.mean(1. - jac)


def sparsity_loss(embedding, hyper_param):
    embedding_mean = embedding.mean(dim=0)
    # Calculate the KL loss for every sample
    losses_of_samples = hyper_param * torch.log(embedding_mean / hyper_param) + \
        (1-hyper_param) * torch.log((1-embedding_mean) / (1-hyper_param))
    return losses_of_samples.sum()


def _map_range_to_01(x):
    """ Returns imput tensor value range to [0, 1] """
    return torch.sigmoid(x)
