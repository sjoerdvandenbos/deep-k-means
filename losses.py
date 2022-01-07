import torch


class MSELoss:
    def __init__(self):
        self.name = "MSE loss"

    @staticmethod
    def __call__(reconstruction, x):
        return torch.square(reconstruction - x).sum(dim=0).mean()

    def __str__(self):
        return self.name


class F1Loss:
    def __init__(self):
        self.name = "F1 loss"

    @staticmethod
    def __call__(x, reconstruction, epsilon=1e-8):
        # Iput range: [0, 1]
        # Reconstruction range: unknown
        reconstruction_binary = _map_range_to_01(reconstruction)
        intersection = torch.sum(x * reconstruction_binary, dim=0)
        denominator = torch.sum(x + reconstruction_binary, dim=0)
        f1 = (2. * intersection + epsilon) / (denominator + epsilon)
        return torch.mean(1. - f1)

    def __str__(self):
        return self.name


class JaccardLoss:
    def __init__(self):
        self.name = "Jaccard loss"

    @staticmethod
    def __call__(x, reconstruction, epsilon=1e-8):
        # Iput range: [0, 1]
        # Reconstruction range: unknown
        reconstruction_binary = _map_range_to_01(reconstruction)
        intersection = x * reconstruction_binary
        union = (x + reconstruction_binary) - intersection
        jac = torch.sum((intersection + epsilon) / (union + epsilon), dim=0)
        return torch.mean(1. - jac)

    def __str__(self):
        return self.name


class CrossEntropyLoss:
    def __init__(self):
        self.name = "Cross Entropy loss"
        self.ce = torch.nn.CrossEntropyLoss()

    def __call__(self, prediction, truth):
        batch_size = prediction.shape[0]
        return self.ce(
            prediction.squeeze().reshape(batch_size, -1),
            truth.long().squeeze().reshape(batch_size, -1)
        )

    def __str__(self):
        return self.name


class BCEWithLogitsLoss:
    def __init__(self):
        self.name = "BCE with logits loss"
        self.bce = torch.nn.BCEWithLogitsLoss()

    @staticmethod
    def __call__(self, prediction, truth):
        return self.bce(prediction, truth)

    def __str__(self):
        return self.name


def sparsity_loss(embedding, hyper_param):
    embedding_mean = embedding.mean(dim=0)
    # Calculate the KL loss for every sample
    losses_of_samples = hyper_param * torch.log(embedding_mean / hyper_param) + \
        (1-hyper_param) * torch.log((1-embedding_mean) / (1-hyper_param))
    return losses_of_samples.sum()


def _map_range_to_01(x):
    """ Returns imput tensor value range to [0, 1] """
    return torch.sigmoid(x)
