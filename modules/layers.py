from torch import nn
import torch
import torch.nn.functional as F


class AvgUnpool2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0, output_padding=0, groups=1,
                 bias=False, device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         output_padding=output_padding, groups=groups, bias=bias)
        avg_factor = 1.0 / torch.tensor(self.kernel_size[0] * self.kernel_size[1], dtype=float)
        # avg_factor = 1.0
        # avg_factor = torch.tensor(self.kernel_size[0] * self.kernel_size[1], dtype=float).item()
        self.weight = nn.Parameter(torch.full(
            size=(out_channels, in_channels // groups, *self.kernel_size),
            fill_value=avg_factor, device=device, dtype=dtype), requires_grad=False
        )


class InterpolateLayer(nn.Module):
    def __init__(self, size, mode="nearest", align_corners=None):
        super().__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return nn.functional.interpolate(x, self.size, mode=self.mode, align_corners=self.align_corners)


class ReshapeLayer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.reshape(*self.args)


class OrthogonalLinearLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.has_bias = bias
        self.n_groups = self._get_number_groups()
        self.group_size = self.out_features // self.n_groups

    def forward(self, x):
        weights = self.weight.view((self.n_groups, self.group_size, self.in_features))
        w_center = weights - weights.mean(dim=-1, keepdim=True)
        del weights
        cov_mat = w_center.bmm(torch.transpose(w_center, dim0=1, dim1=2))
        mappings = self._get_orth_mapping(cov_mat)
        del cov_mat
        new_weights = mappings.bmm(w_center).view((self.out_features, self.in_features))
        del w_center
        return x.mm(new_weights.T) + self.bias if self.has_bias else 0

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


class OrthogonalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        # Conv2d.weight.shape = [out_channels, in_channels, kernel_height, kernel_width]
        self.kernel_height = self.weight.shape[2]
        self.kernel_width = self.weight.shape[3]
        self.has_bias = bias
        self.n_groups = self._get_number_groups()
        self.group_size = self.kernel_height // self.n_groups

    def forward(self, x):
        weights = self.weight.view((-1, self.kernel_height, self.kernel_width))
        w_center = weights - weights.mean(dim=-1, keepdim=True)
        del weights
        cov_mat = w_center.bmm(w_center.transpose(dim0=-2, dim1=-1))
        mappings = self._get_orth_mapping(cov_mat)
        del cov_mat
        new_weights = mappings.bmm(w_center).view((self.out_channels, self.in_channels,
                                                   self.kernel_height, self.kernel_width))
        del w_center
        return F.conv2d(x, new_weights, self.bias, self.stride, self.padding)

    @staticmethod
    def _get_orth_mapping(cov_mat):
        eigen_vecs, eigen_vals, _ = torch.linalg.svd(cov_mat)
        diagonal = eigen_vals.rsqrt().diag_embed()
        return eigen_vecs.bmm(diagonal).bmm(torch.transpose(eigen_vecs, dim0=-2, dim1=-1))

    def _get_number_groups(self):
        group_size = 2
        while self.kernel_height % group_size != 0:
            group_size += 1
        return self.kernel_height // group_size


if __name__ == "__main__":
    pass
