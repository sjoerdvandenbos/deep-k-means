from math import ceil


def get_post_conv_height_and_width(height, width, stride, kernel_size=-1, padding=-1):
    stride = _to_integer(stride)
    post_height = get_post_conv_length(height, stride, kernel_size, padding)
    post_width = get_post_conv_length(width, stride, kernel_size, padding)
    return post_height, post_width


def get_output_padding(pre_conv_height, pre_conv_width, stride):
    stride = _to_integer(stride)
    height_pad = get_output_padding_length(pre_conv_height, stride)
    width_pad = get_output_padding_length(pre_conv_width, stride)
    return height_pad, width_pad


def get_output_padding_length(pre_conv_length, stride):
    if stride == 1:
        return 0
    elif pre_conv_length % stride == 0:
        return 1
    else:
        return 0


def get_post_conv_length(pre_conv_length, stride, kernel_size=-1, padding=-1):
    if kernel_size == -1 and padding == -1:
        return int(ceil(pre_conv_length / stride))
    else:
        return (pre_conv_length + 2*padding - kernel_size) // stride + 1


def _to_integer(potential_tuple):
    if type(potential_tuple) == tuple and potential_tuple[0] == potential_tuple[1]:
        return potential_tuple[0]
    return potential_tuple


if __name__ == "__main__":
    import torch
    before = 299
    print(f"length before: {before}")
    kernel_size = 3
    stride = 2
    padding = 0
    iput = torch.rand(1, 1, before, before)
    post_conv = torch.nn.Conv2d(1, 1, kernel_size, stride, padding)(iput)
    after = get_post_conv_length(before, stride, kernel_size, padding)
    print(f"length after: {after}")
    output_pad = get_output_padding_length(before, stride)
    post_conv_transpose = torch.nn.ConvTranspose2d(
        1, 1, kernel_size, stride, padding, output_padding=output_pad)(post_conv)
    print(f"output padding: {output_pad}")
    print(f"post conv transpose dims: {post_conv_transpose.shape[2:]}")
