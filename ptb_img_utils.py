from PIL import Image
import numpy as np
import tensorflow as tf
import skimage.measure

from glob import glob
from re import compile

# Pattern: any substring of characters only between a leading '_' and trailing '.'
DISEASE_REGEX = compile(r"_[a-zA-z]*\.")
DISEASE_MAPPING = {
    "BundleBranchBlock": 0,
    "Cardiomyopathy": 1,
    "Dysrhythmia": 2,
    "HealthyControl": 3,
    "MyocardialInfarction": 4,
    "Myocarditis": 5,
    "ValvularHeartDisease": 6
}


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
    # Reduce pixel range to [0, 1]
    mapped = cropped / 255.0
    reduced = skimage.measure.block_reduce(mapped, (7, 7), np.min)
    return reduced.flatten()


if __name__ == "__main__":
    from pathlib import Path
    data_path = Path.cwd() / "split" / "ptb-images-2-cropped"
    print("Loading data into memory...", flush=True)
    filenames, diseases = get_filenames_and_labels(data_path)
    data = np.asarray([get_image_tensor(f) for f in filenames])
    print("Writing to disk...", flush=True)
    np.save(data_path / "compacted_data.npy", data)
    np.save(data_path / "compacted_target.npy", diseases)
    print("Done!", flush=True)

