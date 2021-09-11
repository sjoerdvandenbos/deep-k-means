from PIL import Image
import numpy as np
import tensorflow as tf
import skimage.measure

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
    batch_data = np.array([get_image_tensor(p).flatten() for p in batch_paths])
    return batch_indices, batch_data


def get_filenames_and_labels(dataset_path: str) -> (np.ndarray, np.ndarray):
    files = sorted(glob(f"{dataset_path}/*/*/*"))
    array = np.array(files)
    labels = np.array([get_label(f) for f in files])
    return array, labels


def get_label(path: str) -> str:
    """ Turns the file path of an image into the correct label for that image. """
    matched = DISEASE_REGEX.search(path).group(0)
    # Get rid of the leading '_' and trailing '.'
    disease = matched[1:-1]
    return disease


def get_image_tensor(path: str) -> np.ndarray:
    """ Takes the input file path and returns a 2D Tensor
    containing all pixel values of a single channel. """
    image = Image.open(path)
    # Image is black and white, so one channel will give all info needed
    red, _, _ = image.split()
    cropped = np.array(red)[11:2209, 15:2703]
    # Reduce pixel range to [0, 1]
    mapped = cropped / 255.0
    reduced = skimage.measure.block_reduce(mapped, (7, 7), np.min)
    return reduced


def memory():
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    print('memory use:', memoryUse)


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    path_string = "split/ptb-images-2-cropped/train/ValvularHeartDisease/patient106_s0030_re_1_ValvularHeartDisease.jpg"
    file_names, targets = get_filenames_and_labels()
    idx, mini_batch = next_batch(2, file_names)
    values = mini_batch.numpy()
    s = set(values.flatten())
    print(s)
