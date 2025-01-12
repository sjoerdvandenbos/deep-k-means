import torch
from torch import nn

from modules.utils import get_output_padding


class BottleneckConvBlock(nn.Module):
    """ Implementation from github.com/JayPatwardhan/ResNet-PyTorch """
    expansion = 4

    def __init__(self, in_channels, out_channels, down_sampler=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        # self.dropout1 = nn.Dropout2d(p=0.1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        # self.dropout2 = nn.Dropout2d(p=0.1)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.down_sampler = down_sampler
        self.stride = stride
        self.relu = nn.ReLU()
        # self.dropout3 = nn.Dropout2d(p=0.1)

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)
        # downsample if needed
        if self.down_sampler is not None:
            identity = self.down_sampler(identity)
        # add identity
        x += identity
        x = self.relu(x)
        return x


class BottleneckConvTransposeBlock(nn.Module):
    expansion = 4

    def __init__(self, post_conv_transpose_height, post_conv_transpose_width, in_channels, out_channels,
                 up_sampler=None, stride=1, last=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height, self.width = post_conv_transpose_height, post_conv_transpose_width
        output_pad_height, output_pad_width = get_output_padding(post_conv_transpose_height,
                                                                 post_conv_transpose_width, stride)
        self.conv3 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        # self.dropout3 = nn.Dropout2d(p=0.1)
        self.batch_norm3 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=stride,
                                        padding=1, output_padding=(output_pad_height, output_pad_width), bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        # self.dropout2 = nn.Dropout2d(p=0.1)
        if last:
            self.conv1 = nn.ConvTranspose2d(out_channels, out_channels*stride, kernel_size=1, stride=1, padding=0,
                                            bias=True)
            self.batch_norm1 = nn.BatchNorm2d(out_channels*stride)
        else:
            self.conv1 = nn.ConvTranspose2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0,
                                            bias=True)
            self.batch_norm1 = nn.BatchNorm2d(out_channels*self.expansion)

        self.up_sampler = up_sampler
        self.stride = stride
        self.relu = nn.ReLU()
        # self.dropout1 = nn.Dropout2d(p=0.1)

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm3(self.conv3(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm1(self.conv1(x))
        if self.up_sampler is not None:
            identity = self.up_sampler(identity)
        x = x + identity
        x = self.relu(x)
        return x

    def __repr__(self):
        layer_info = super().__repr__()
        block_info = f"in_channels={self.in_channels}, out_channels={self.out_channels}"
        return f"{block_info}\n{layer_info}"


class BasicConvBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, down_sampler=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        # self.dropout1 = nn.Dropout2d(p=0.1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        # self.dropout2 = nn.Dropout2d(p=0.1)
        self.down_sampler = down_sampler
        # self.dropout3 = nn.Dropout2d(p=0.1)

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        if self.down_sampler is not None:
            identity = self.down_sampler(identity)
        x = x + identity
        x = self.relu(x)
        return x


class BasicConvTransposeBlock(nn.Module):
    expansion = 1

    def __init__(self, post_conv_transpose_height, post_conv_transpose_width, in_channels, out_channels,
                 up_sampler=None, stride=1, last=False):
        super().__init__()
        output_padding = get_output_padding(post_conv_transpose_height, post_conv_transpose_width, stride)
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                                                  output_padding=output_padding, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        # self.dropout1 = nn.Dropout2d(p=0.1)
        self.conv_transpose2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                                  bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        # self.dropout2 = nn.Dropout2d(p=0.1)
        self.up_sampler = up_sampler
        # self.dropout3 = nn.Dropout2d(p=0.1)

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv_transpose1(x)))
        x = self.relu(self.batch_norm2(self.conv_transpose2(x)))
        if self.up_sampler is not None:
            identity = self.up_sampler(identity)
        x = x + identity
        x = self.relu(x)
        return x


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=True, batch_norm=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else None
        self.relu = nn.ReLU() if activation else None

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x) if self.batch_norm is not None else x
        x = self.relu(x) if self.relu is not None else x
        return x


class ConvTransposeBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding,
                 activation=True, batch_norm=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                                 output_padding=output_padding)
        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else None
        self.relu = nn.ReLU() if activation else None

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x) if self.batch_norm is not None else x
        x = self.relu(x) if self.relu is not None else x
        return x


class Conv1dHyperbolicTangent(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout_probability=0.5,
                 bn=True, activation=True):
        super().__init__()
        self.activation_enabled = activation
        self.bn_enabled = bn
        bias = not bn
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) if self.bn_enabled else x
        x = torch.tanh(x) if self.activation_enabled else x
        x = self.dropout(x)
        return x


class Conv1dTransposeHyperbolicTangent(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout_probability=0.5,
                 bn=True, activation=True):
        super().__init__()
        self.activation_enabled = activation
        self.bn_enabled = bn
        bias = not bn
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) if self.bn_enabled else x
        x = torch.tanh(x) if self.activation_enabled else x
        x = self.dropout(x)
        return x


class ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 batch_norm=True, activation=True):
        super().__init__()
        bias = not batch_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.relu = nn.ReLU() if activation else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) if self.bn is not None else x
        x = self.relu(x) if self.relu is not None else x
        return x


class ConvTransposeBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0,
                 batch_norm=True, activation=True):
        super().__init__()
        bias = not batch_norm
        self.conv_tranpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias,
                                                output_padding=output_padding
                                                )
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.relu = nn.ReLU() if activation else None

    def forward(self, x):
        x = self.conv_tranpose(x)
        x = self.bn(x) if self.bn is not None else x
        x = self.relu(x) if self.relu is not None else x
        return x



