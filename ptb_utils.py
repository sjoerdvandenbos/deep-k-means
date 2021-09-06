from PIL import Image
import numpy as np
import tensorflow as tf

from glob import glob
from re import compile

# Pattern: any substring of characters only between a leading '_' and trailing '.'
DISEASE_REGEX = compile(r"_[a-zA-z]*\.")


def next_batch(num: tf.int32, data: np.ndarray) -> (np.ndarray, tf.Tensor):
    """ Returns a batch of num random data """
    # Generate random indices to load for this minibatch.
    indices = np.arange(0, data.shape[0], dtype=np.int32)
    np.random.shuffle(indices)
    batch_indices = indices[:num]
    batch_paths = data[batch_indices]
    # Shape of batch_data = [N, H, W]
    batch_data = tf.convert_to_tensor([get_image_tensor(p) for p in batch_paths])
    shape = batch_data.shape
    # Reshape to [N, H, W, C]
    batch_data = tf.reshape(batch_data, [shape[0], shape[1], shape[2], 1])
    down_sampled = tf.nn.conv2d(batch_data,
                                [[[[1]]]],
                                padding="VALID",
                                strides=[7, 7],
                                data_format="NHWC")
    flat = tf.reshape(down_sampled, [num, -1])
    return batch_indices, flat.eval()


def get_filenames_and_labels() -> (np.ndarray, np.ndarray):
    files = sorted(glob(r"split/ptb-images-2-cropped/*/*/*.jpg"))
    array = np.array(files)
    labels = np.array([get_label(f) for f in files])
    return array, labels


def get_label(path: str) -> str:
    """ Turns the file path of an image into the correct label for that image. """
    matched = DISEASE_REGEX.search(path).group(0)
    # Get rid of the leading '_' and trailing '.'
    disease = matched[1:-1]
    return disease


def get_image_tensor(path: str) -> tf.Tensor:
    """ Takes the input file path and returns a 2D Tensor
    containing all pixel values of a single channel. """
    image = Image.open(path)
    # Image is black and white, so one channel will give all info needed
    red, _, _ = image.split()
    cropped = tf.convert_to_tensor(np.array(red), dtype=tf.float32)[11:2209, 15:2703]
    # Reduce pixel range to [0, 1]
    reduced = cropped / 255.0
    return reduced


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    path_string = "split/ptb-images-2-cropped/train/ValvularHeartDisease/patient106_s0030_re_1_ValvularHeartDisease.jpg"
    file_names, targets = get_filenames_and_labels()
    idx, mini_batch = next_batch(2, file_names)
    values = mini_batch.numpy()
    s = set(values.flatten())
    print(s)
