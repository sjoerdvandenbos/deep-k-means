from torch import nn

from modules.blocks import Conv1dHyperbolicTangent, Conv1dTransposeHyperbolicTangent
from modules.layers import ReshapeLayer


class Base1dConvEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.simple_layers = nn.Sequential(
            Conv1dHyperbolicTangent(in_channels, 10, 7, 1, 3),
            nn.MaxPool1d(2),
            Conv1dHyperbolicTangent(10, 32, 6, 1, 3),
            nn.MaxPool1d(2),
            Conv1dHyperbolicTangent(32, 64, 5, 1, 3),
            nn.MaxPool1d(2),
            Conv1dHyperbolicTangent(64, 64, 7, 1, 3, bn=False),
            Conv1dHyperbolicTangent(64, 64, 11, 1, 5),
            nn.MaxPool1d(2),
            Conv1dHyperbolicTangent(64, 128, 11, 1, 5, bn=False),
            Conv1dHyperbolicTangent(128, 64, 11, 1, 5),
            nn.MaxPool1d(2),
            Conv1dHyperbolicTangent(64, 1, 32, 1, 16),
        )

    def forward(self, x):
        x = self.simple_layers(x)
        return x


class Simple1dConvEncoder(nn.Module):
    def __init__(self, signal_length, in_channels, embedding_size):
        super().__init__()
        self.linear_layers = Base1dConvEncoder(in_channels)
        post_linear_layers_length = signal_length // 32
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(post_linear_layers_length, embedding_size)

    def forward(self, x):
        x = self.linear_layers(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class Complex1dConvEncoder(nn.Module):
    def __init__(self, in_channels, embedding_size):
        super().__init__()
        self.linear_layers = Base1dConvEncoder(in_channels)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=1,
            num_layers=8,
            batch_first=True,
        )
        self.linear = nn.Linear(8, embedding_size)

    def forward(self, x):
        x = self.linear_layers(x)
        # x.shape = [N, C, L]   C=1
        # lstm_format.shape = [N, L, Feature]   Feature=1
        x = x.permute(0, 2, 1)
        _, (x, _) = self.lstm(x)
        x = self.linear(x)
        return x


class Base1dConvDecoder(nn.Module):
    def __init__(self, in_features, out_length, out_channels):
        super().__init__()
        self.simple_layers = nn.Sequential(
            ReshapeLayer(-1, 1, 32),
            Conv1dTransposeHyperbolicTangent(1, 64, 32, 1, 16),
            nn.Upsample(scale_factor=2),
            Conv1dTransposeHyperbolicTangent(64, 128, 11, 1, 5, bn=False),
            Conv1dTransposeHyperbolicTangent(128, 64, 11, 1, 5),
            nn.Upsample(scale_factor=2),
            Conv1dTransposeHyperbolicTangent(64, 64, 11, 1, 5, bn=False),
            Conv1dTransposeHyperbolicTangent(64, 64, 7, 1, 3),
            nn.Upsample(scale_factor=2),
            Conv1dTransposeHyperbolicTangent(64, 32, 5, 1, 3),
            nn.Upsample(scale_factor=2),
            Conv1dTransposeHyperbolicTangent(32, 10, 6, 1, 3),
            nn.Upsample(scale_factor=2),
            Conv1dTransposeHyperbolicTangent(10, out_channels, 7, 1, 3, bn=False, activation=False),
        )

