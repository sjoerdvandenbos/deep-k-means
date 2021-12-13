from math import ceil


def get_post_conv_height_and_width(height, width, stride):
    stride = _to_integer(stride)
    post_height = _get_post_conv_length(height, stride)
    post_width = _get_post_conv_length(width, stride)
    return post_height, post_width


def get_output_padding(pre_conv_height, pre_conv_width, stride):
    if type(stride) == tuple:
        stride = stride[0]
    height_pad = _get_output_padding_length(pre_conv_height, stride)
    width_pad = _get_output_padding_length(pre_conv_width, stride)
    return height_pad, width_pad


def _get_output_padding_length(pre_conv_length, stride):
    if stride == 1:
        return 0
    elif pre_conv_length % stride == 0:
        return 1
    else:
        return 0


def _get_post_conv_length(pre_conv_length, stride):
    return int(ceil(pre_conv_length / stride))


def _to_integer(potential_tuple):
    if type(potential_tuple) == tuple:
        return potential_tuple[0]
    return potential_tuple


if __name__ == "__main__":
    before = 6
    after = _get_post_conv_length(before, 2)
    print(f"length after: {after}")
    output_pad = _get_output_padding_length(before)
    print(f"output padding: {output_pad}")
