import torch
from PIL import Image
import numpy as np
import skimage.measure
from matplotlib import cm

from glob import glob
from re import compile
from pathlib import Path

# Pattern: any substring of characters only between a leading '_' and trailing '.'
DISEASE_REGEX = compile(r"_[a-zA-z]*\.")


def next_batch(num: int, data: np.ndarray) -> (np.ndarray, np.ndarray):
    """ Returns a batch of num random data """
    # Generate random indices to load for this minibatch.
    indices = np.arange(0, data.shape[0], dtype=np.int32)
    np.random.shuffle(indices)
    batch_indices = indices[:num]
    batch_paths = data[batch_indices]
    batch_data = np.array([get_image_tensor(p) for p in batch_paths])
    return batch_indices, batch_data


def get_filenames_and_labels(dataset_path) -> (np.ndarray, np.ndarray):
    if type(dataset_path) == str:
        files = sorted(glob(f"{dataset_path}/*/*/*"))
    else:
        files = sorted(dataset_path.glob("*/*/*"))
    array = np.array(files)
    labels = np.asarray([get_label(f) for f in files])
    return array, labels


def get_label(path) -> str:
    """ Turns the file path of an image into the correct label for that image. """
    matched = DISEASE_REGEX.search(str(path)).group(0)
    # Get rid of the leading '_' and trailing '.'
    disease = matched[1:-1]
    return disease


def get_image_tensor(path) -> np.ndarray:
    """ Takes the input file path and returns a 2D Tensor
    containing all pixel values of a single channel. """
    image = Image.open(path)
    # Image is black and white, so one channel will give all info needed
    red, _, _ = image.split()
    cropped = np.array(red)[11:2209, 15:2703]
    reduced = skimage.measure.block_reduce(cropped, (7, 7), np.min)
    inverted = 255 - reduced
    return inverted.astype(np.uint8)


def save_imgs_to_npy(path):
    data_path = Path.cwd() / path
    print("Loading data into memory...", flush=True)
    filenames, diseases = get_filenames_and_labels(data_path)
    data = np.array([get_image_tensor(f) for f in filenames], dtype=np.uint8)
    print("Writing to disk...", flush=True)
    np.save(path / "compacted_data", data)
    np.save(path / "compacted_target", diseases)
    print("Done!", flush=True)


def reconstruct_image(pixel_vector: np.ndarray, shape, transform=False):
    """" Returns 2d numpy array with pixel vals distributed N(0.5; 255/2) """
    numpy_right_shape = pixel_vector.reshape(shape)
    if transform:
        mean = np.mean(numpy_right_shape)
        stddev = np.std(numpy_right_shape)
        transformed = ((numpy_right_shape - mean) / (4. * stddev) + 0.5)
        return Image.fromarray(np.uint8(cm.gray(transformed) * 255))
    else:
        return Image.fromarray(np.uint8(cm.gray(numpy_right_shape) * 255))


if __name__ == "__main__":
    save_imgs_to_npy(Path.cwd() / "split" / "ptb-images-2-cropped")
