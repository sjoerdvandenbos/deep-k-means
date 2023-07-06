import torch
from torch import nn

from modules.blocks import ConvBlock2d, ConvTransposeBlock2d
from modules.layers import InterpolateLayer, ReshapeLayer
from modules.utils import get_output_padding, get_post_conv_height_and_width

RESIDUAL_SCALING_FACTOR = 0.17


class InceptionResNetV2(nn.Module):
    def __init__(self, height, width, embedding_size, n_channels):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool_indices_stack = []

        stem = Stem(height, width, n_channels)
        post_stem_size = stem.get_output_size()
        self.encoder.append(stem)
        for _ in range(5):
            self.encoder.append(InceptionResNetAV2(384, 384))
        ra = InceptionResNetV2ReductionA(384)
        post_reduction_a_channels = ra.out_channels
        post_reduction_a_size = InceptionResNetV2ReductionA.get_output_size(post_stem_size)
        self.encoder.append(ra)
        for _ in range(10):
            self.encoder.append(InceptionResNetBV2(post_reduction_a_channels, post_reduction_a_channels))
        rb = InceptionResNetV2ReductionB(post_reduction_a_channels)
        post_reduction_b_channels = rb.out_channels
        post_reduction_b_size = InceptionResNetV2ReductionB.get_output_size(post_reduction_a_size)
        self.encoder.append(rb)
        for _ in range(5):
            self.encoder.append(InceptionResnetCV2(post_reduction_b_channels, post_reduction_b_channels))
        self.encoder.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.encoder.append(nn.Flatten())
        self.encoder.append(nn.Linear(post_reduction_b_channels, embedding_size))

        self.decoder.append(nn.Linear(embedding_size, post_reduction_b_channels))
        self.decoder.append(ReshapeLayer(-1, post_reduction_b_channels, 1, 1))
        self.decoder.append(InterpolateLayer(post_reduction_b_size))
        for _ in range(5):
            self.decoder.append(InceptionResNetCV2Reversed(post_reduction_b_channels, post_reduction_b_channels))
        self.decoder.append(InceptionResNetV2ReductionBReversed(post_reduction_a_channels, post_reduction_a_size))
        for _ in range(10):
            self.decoder.append(InceptionResNetBV2Reversed(post_reduction_a_channels, post_reduction_a_channels))
        self.decoder.append(InceptionResNetV2ReductionAReversed(384, post_stem_size))
        for _ in range(5):
            self.decoder.append(InceptionResNetAV2Reversed(384, 384))
        self.decoder.append(StemReversed(stem))

    def forward(self, x):
        x = self.forward_encoder(x)
        embeds = x
        x = self.forward_decoder(x)
        return embeds, x

    def forward_encoder(self, x):
        for module in self.encoder:
            if type(module) in (Stem, InceptionResNetV2ReductionA, InceptionResNetV2ReductionB):
                x, indices = module(x)
                self.pool_indices_stack.append(indices)
            else:
                x = module(x)
        return x

    def forward_decoder(self, x):
        for module in self.decoder:
            if type(module) in (StemReversed, InceptionResNetV2ReductionAReversed, InceptionResNetV2ReductionBReversed):
                indices = self.pool_indices_stack.pop()
                x = module(x, indices)
            else:
                x = module(x)
        return x


class Stem(nn.Module):
    def __init__(self, height, width, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self._set_post_conv_dims(height, width)
        self.conv_seq = nn.Sequential(
            ConvBlock2d(in_channels, 32, kernel_size=3, stride=2, padding=0),
            ConvBlock2d(32,          32, kernel_size=3, stride=1, padding=0),
            ConvBlock2d(32,          64, kernel_size=3, stride=1, padding=1),
        )
        self.conv1 = ConvBlock2d(64, 96, kernel_size=3, stride=2, padding=0)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, return_indices=True)
        self.conv_seq_short = nn.Sequential(
            ConvBlock2d(160, 64, kernel_size=1, stride=1, padding=0),
            ConvBlock2d(64, 96, kernel_size=3, stride=1, padding=0),
        )
        self.conv_seq_long = nn.Sequential(
            ConvBlock2d(160, 64, kernel_size=1, stride=1, padding=0),
            ConvBlock2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            ConvBlock2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock2d(64, 96, kernel_size=3, stride=1, padding=0),
        )
        self.conv2 = ConvBlock2d(192, 192, kernel_size=3, stride=2, padding=0)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, return_indices=True)

    def forward(self, x):
        x = self.conv_seq(x)
        x2 = self.conv1(x)
        x, indices1 = self.max_pool1(x)
        x = torch.cat((x, x2), dim=1)
        x2 = self.conv_seq_long(x)
        x = self.conv_seq_short(x)
        x = torch.cat((x, x2), dim=1)
        x2 = self.conv2(x)
        x, indices2 = self.max_pool2(x)
        x = torch.cat((x, x2), dim=1)
        return x, (indices1, indices2)

    def _set_post_conv_dims(self, height, width):
        self.height = height
        self.width = width
        self.height1, self.width1 = get_post_conv_height_and_width(height, width, stride=2, kernel_size=3, padding=0)
        self.height2, self.width2 = get_post_conv_height_and_width(
            self.height1, self.width1, stride=1, kernel_size=3, padding=0)
        self.height3, self.width3 = get_post_conv_height_and_width(
            self.height2, self.width2, stride=2, kernel_size=3, padding=0)
        self.height4, self.width4 = get_post_conv_height_and_width(
            self.height3, self.width3, stride=1, kernel_size=3, padding=0)
        self.height5, self.width5 = get_post_conv_height_and_width(
            self.height4, self.width4, stride=2, kernel_size=3, padding=0)

    def get_output_size(self):
        return self.height5, self.width5


class StemReversed(nn.Module):
    def __init__(self, stem):
        super().__init__()
        self.output_channels = stem.in_channels
        self._set_output_paddings(stem)
        self.conv_2 = ConvTransposeBlock2d(192, 192, kernel_size=3, stride=2, padding=0,
                                           output_padding=self.output_pad4)
        self.maxunpool2 = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=0)
        self.seq_short = nn.Sequential(
            ConvTransposeBlock2d(96, 64, kernel_size=3, stride=1, padding=0, output_padding=self.output_pad3),
            ConvBlock2d(64, 160, kernel_size=1, stride=1, padding=0),
        )
        self.seq_long = nn.Sequential(
            ConvTransposeBlock2d(96, 64, kernel_size=3, stride=1, padding=0, output_padding=self.output_pad3),
            ConvBlock2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            ConvBlock2d(64, 160, kernel_size=1, stride=1, padding=0),
        )
        self.conv_1 = ConvTransposeBlock2d(96, 64, kernel_size=3, stride=2, padding=0, output_padding=self.output_pad2)
        self.maxunpool1 = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=0)
        self.conv_seq_1 = nn.Sequential(
            ConvBlock2d(64, 32, kernel_size=3, stride=1, padding=1),
            ConvTransposeBlock2d(32, 32, kernel_size=3, stride=1, padding=0, output_padding=self.output_pad1),
            ConvTransposeBlock2d(32, self.output_channels, kernel_size=3, stride=2, padding=0,
                                 output_padding=self.output_pad),
        )

    def forward(self, x, unpool_indices):
        x, x2 = self._divide_in_2_over_channels(x)
        x2 = self.conv_2(x2)
        x = self.maxunpool1(x, unpool_indices[1], output_size=self.dims4)
        x = x + x2  # Reversed operation of a branch out.
        x, x2 = self._divide_in_2_over_channels(x)
        x2 = self.seq_short(x2)
        x = self.seq_long(x)
        x = x + x2  # Reversed operation of a branch out.
        x, x2 = self._divide_in_2_over_channels(x, border=96)
        x = self.conv_1(x)
        x2 = self.maxunpool1(x2, unpool_indices[0], output_size=self.dims2)
        x = x + x2  # Reversed operation of a branch out.
        x = self.conv_seq_1(x)
        return x

    @staticmethod
    def _divide_in_2_over_channels(x, border=None):
        """" Reversed operation of filter concat. """
        if border is None:
            n_channels = x.shape[1]
            halfway = n_channels // 2
            border = halfway
        return (
            x[:, :border, :, :],
            x[:, border:, :, :]
        )

    def _set_output_paddings(self, stem):
        self.output_pad = get_output_padding(stem.height, stem.width, stride=2)
        self.output_pad1 = get_output_padding(stem.height1, stem.width1, stride=1)
        self.output_pad2 = get_output_padding(stem.height2, stem.width2, stride=2)
        self.output_pad3 = get_output_padding(stem.height3, stem.width3, stride=1)
        self.output_pad4 = get_output_padding(stem.height4, stem.width4, stride=2)
        self.dims2 = stem.height2, stem.width2
        self.dims4 = stem.height4, stem.width4


class InceptionResNetAV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = ConvBlock2d(in_channels, 32, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Sequential(
            ConvBlock2d(in_channels, 32, kernel_size=1, stride=1, padding=0),
            ConvBlock2d(32, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBlock2d(in_channels, 32, kernel_size=1, stride=1, padding=0),
            ConvBlock2d(32, 48, kernel_size=3, stride=1, padding=1),
            ConvBlock2d(48, 64, kernel_size=3, stride=1, padding=1),
        )
        self.unified_conv = ConvBlock2d(128, out_channels, kernel_size=1, stride=1, padding=0, activation=False)
        self.residual_conn = ConvBlock2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                         activation=False, batch_norm=False
                                         ) if in_channels != out_channels else None

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.residual_conn(x) if self.residual_conn is not None else x
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.unified_conv(x)
        return x + RESIDUAL_SCALING_FACTOR*x4


class InceptionResNetAV2Reversed(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = ConvBlock2d(32, out_channels, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Sequential(
            ConvBlock2d(32, 32, kernel_size=3, stride=1, padding=1),
            ConvBlock2d(32, out_channels, kernel_size=1, stride=1, padding=0),
        )
        self.branch3 = nn.Sequential(
            ConvBlock2d(64, 48, kernel_size=3, stride=1, padding=1),
            ConvBlock2d(48, 32, kernel_size=3, stride=1, padding=1),
            ConvBlock2d(32, out_channels, kernel_size=1, stride=1, padding=0),
        )
        self.splitting_conv = ConvBlock2d(in_channels, 128, kernel_size=1, stride=1, padding=0, activation=False)
        self.residual_conn = ConvBlock2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                         activation=False, batch_norm=False
                                         ) if in_channels != out_channels else None

    def forward(self, x):
        x1 = self.splitting_conv(x)
        x2 = x1[:, :32, :, :]
        x3 = x1[:, 32:64, :, :]
        x4 = x1[:, 64:, :, :]
        x2 = self.branch1(x2)
        x3 = self.branch2(x3)
        x4 = self.branch3(x4)
        x = self.residual_conn(x) if self.residual_conn is not None else x
        return RESIDUAL_SCALING_FACTOR*x + x2 + x3 + x4


class InceptionResNetBV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = ConvBlock2d(in_channels, 192, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Sequential(
            ConvBlock2d(in_channels, 128, kernel_size=1, stride=1, padding=0),
            ConvBlock2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)),
        )
        self.unifying_conv = ConvBlock2d(384, out_channels, kernel_size=1, stride=1, padding=0,
                                         activation=False, batch_norm=False)
        self.residual_conn = ConvBlock2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                         activation=False, batch_norm=False,
                                         ) if in_channels != out_channels else None

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.unifying_conv(x1)
        x = self.residual_conn(x) if self.residual_conn is not None else x
        x = RESIDUAL_SCALING_FACTOR*x + x1
        return x


class InceptionResNetBV2Reversed(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.splitting_conv = ConvBlock2d(in_channels, 384, kernel_size=1, stride=1, padding=0,
                                          activation=False)
        self.branch1 = ConvBlock2d(192, out_channels, kernel_size=1, stride=1, padding=0, batch_norm=False)
        self.branch2 = nn.Sequential(
            ConvBlock2d(192, 160, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            ConvBlock2d(160, 128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock2d(128, out_channels, kernel_size=1, stride=1, padding=0, batch_norm=False),
        )
        self.residual_conn = ConvBlock2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                         activation=False, batch_norm=False
                                         ) if in_channels != out_channels else None

    def forward(self, x):
        x1 = self.splitting_conv(x)
        x2 = x1[:, :192, :, :]
        x3 = x1[:, 192:, :, :]
        x2 = self.branch1(x2)
        x3 = self.branch2(x3)
        x = self.residual_conn(x) if self.residual_conn is not None else x
        x = RESIDUAL_SCALING_FACTOR*x + x2 + x3
        return x


class InceptionResnetCV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = ConvBlock2d(in_channels, 192, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Sequential(
            ConvBlock2d(in_channels, 192, kernel_size=1, stride=1, padding=0),
            ConvBlock2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            ConvBlock2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        )
        self.unifying_conv = ConvBlock2d(448, out_channels, kernel_size=1, stride=1, padding=0,
                                         activation=False, batch_norm=False)
        self.residual_conn = ConvBlock2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                         activation=False, batch_norm=False
                                         ) if in_channels != out_channels else None

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.unifying_conv(x1)
        x = self.residual_conn(x) if self.residual_conn is not None else x
        x = RESIDUAL_SCALING_FACTOR*x + x1
        return x


class InceptionResNetCV2Reversed(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.splitting_conv = ConvBlock2d(in_channels, 448, kernel_size=1, stride=1, padding=0, activation=False)
        self.branch1 = ConvBlock2d(192, out_channels, kernel_size=1, stride=1, padding=0, batch_norm=False)
        self.branch2 = nn.Sequential(
            ConvBlock2d(256, 224, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            ConvBlock2d(224, 192, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            ConvBlock2d(192, out_channels, kernel_size=1, stride=1, padding=0, batch_norm=False),
        )
        self.residual_conn = ConvBlock2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                         activation=False, batch_norm=False
                                         ) if in_channels != out_channels else None

    def forward(self, x):
        x1 = self.splitting_conv(x)
        x2 = x1[:, :192, :, :]
        x3 = x1[:, 192:, :, :]
        x2 = self.branch1(x2)
        x3 = self.branch2(x3)
        x = self.residual_conn(x) if self.residual_conn is not None else x
        x = RESIDUAL_SCALING_FACTOR*x + x2 + x3
        return x


class InceptionResNetV2ReductionA(nn.Module):
    def __init__(self, in_channels, n=384, k=256, l=256, m=384):
        super().__init__()
        self.out_channels = in_channels + n + m
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, return_indices=True)
        self.branch1 = ConvBlock2d(in_channels, n, kernel_size=3, stride=2, padding=0)
        self.branch2 = nn.Sequential(
            ConvBlock2d(in_channels, k, kernel_size=1, stride=1, padding=0),
            ConvBlock2d(k, l, kernel_size=3, stride=1, padding=1),
            ConvBlock2d(l, m, kernel_size=3, stride=2, padding=0),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x, indices = self.maxpool(x)
        x = torch.cat((x, x1, x2), dim=1)
        return x, indices

    @staticmethod
    def get_output_size(input_size):
        return get_post_conv_height_and_width(*input_size, stride=2, kernel_size=3, padding=0)


class InceptionResNetV2ReductionAReversed(nn.Module):
    def __init__(self, out_channels, output_size, n=384, k=256, l=256, m=384):
        super().__init__()
        self.n = n
        self.out_channels = out_channels
        self.output_size = output_size
        output_padding = get_output_padding(*output_size, stride=2)
        self.maxunpool = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=0)
        self.branch1 = ConvTransposeBlock2d(n, out_channels, kernel_size=3, stride=2, padding=0,
                                            output_padding=output_padding)
        self.branch2 = nn.Sequential(
            ConvTransposeBlock2d(m, l, kernel_size=3, stride=2, padding=0, output_padding=output_padding),
            ConvBlock2d(l, k, kernel_size=3, stride=1, padding=1),
            ConvBlock2d(k, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, indices):
        border = self.out_channels + self.n
        x1 = x[:, self.out_channels:border, :, :]
        x2 = x[:, border:, :, :]
        x = x[:, :self.out_channels, :, :]
        x = self.maxunpool(x, indices, output_size=self.output_size)
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        x = x + x1 + x2
        return x


class InceptionResNetV2ReductionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.out_channels = in_channels + 384 + 288 + 320
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, return_indices=True)
        self.branch1 = nn.Sequential(
            ConvBlock2d(in_channels, 256, kernel_size=1, stride=1, padding=0),
            ConvBlock2d(256, 384, kernel_size=3, stride=2, padding=0),
        )
        self.branch2 = nn.Sequential(
            ConvBlock2d(in_channels, 256, kernel_size=1, stride=1, padding=0),
            ConvBlock2d(256, 288, kernel_size=3, stride=2, padding=0),
        )
        self.branch3 = nn.Sequential(
            ConvBlock2d(in_channels, 256, kernel_size=1, stride=1, padding=0),
            ConvBlock2d(256, 288, kernel_size=3, stride=1, padding=1),
            ConvBlock2d(288, 320, kernel_size=3, stride=2, padding=0),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x, indices = self.maxpool(x)
        x = torch.cat((x, x1, x2, x3), dim=1)
        return x, indices

    @staticmethod
    def get_output_size(input_size):
        return get_post_conv_height_and_width(*input_size, stride=2, kernel_size=3, padding=0)


class InceptionResNetV2ReductionBReversed(nn.Module):
    def __init__(self, out_channels, output_size):
        super().__init__()
        self.out_channels = out_channels
        self.output_size = output_size
        output_padding = get_output_padding(*output_size, stride=2)
        self.maxunpool = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=0)
        self.channels1 = 384
        self.branch1 = nn.Sequential(
            ConvTransposeBlock2d(self.channels1, 256, kernel_size=3, stride=2, padding=0, output_padding=output_padding),
            ConvBlock2d(256, out_channels, kernel_size=1, stride=1, padding=0),
        )
        self.channels2 = 288
        self.branch2 = nn.Sequential(
            ConvTransposeBlock2d(self.channels2, 256, kernel_size=3, stride=2, padding=0, output_padding=output_padding),
            ConvBlock2d(256, out_channels, kernel_size=1, stride=1, padding=0),
        )
        self.channels3 = 320
        self.branch3 = nn.Sequential(
            ConvTransposeBlock2d(self.channels3, 288, kernel_size=3, stride=2, padding=0, output_padding=output_padding),
            ConvBlock2d(288, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock2d(256, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, indices):
        border1 = self.out_channels
        border2 = border1 + self.channels1
        border3 = border2 + self.channels2
        x1 = x[:, border1:border2, :, :]
        x2 = x[:, border2:border3, :, :]
        x3 = x[:, border3:, :, :]
        x = x[:, :border1, :, :]
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        x3 = self.branch3(x3)
        x = self.maxunpool(x, indices, output_size=self.output_size)
        x = x1 + x2 + x3 + x
        return x


if __name__ == "__main__":
    iput = torch.rand(1, 21, 299, 299)
    _, c, h, w = iput.shape
    # encoder = Stem(h, w, c)
    # encoder = InceptionResnetAv2(c, out_c)
    encoder = InceptionResNetV2ReductionB(c)
    # embedding = encoder(iput)
    embeds, i = encoder(iput)
    print(f"embeds shape: {embeds.shape}")
    print(f"predicted embeds channels: {encoder.out_channels}")
    print(f"predicted embeds size: {encoder.get_output_size((h, w))}")
    # decoder = StemReversed(encoder)
    # decoder = InceptionResenetAv2Reversed(out_c, c)
    decoder = InceptionResNetV2ReductionBReversed(c, (h, w))
    # reconstruction = decoder(embedding)
    # reconstruction = decoder(embedding, i)
    reconstruction = decoder(embeds, i)
    print(reconstruction.shape)