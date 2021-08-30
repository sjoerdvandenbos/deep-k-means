from PIL import Image
import numpy as np
import pandas as pd

from glob import glob
from re import compile

REGEX_MACHINE = compile(r"_[a-zA-z]*\.")


def next_batch(num: int, data: np.array) -> (list, np.ndarray):
    """ Returns a batch of num random data """
    indices = np.arange(0, data.shape[0])
    np.random.shuffle(indices)
    batch_indices = indices[:num]
    batch_paths = data[batch_indices]
    batch_data = np.asarray([get_pixel_series(p) for p in batch_paths])
    return indices, batch_data


def get_filenames_and_labels() -> (list, list):
    files = sorted(glob(r"split/ptb-images-2-cropped/*/*/*.jpg"))
    labels = np.array([get_label(f) for f in files])
    return files, labels


def get_label(path: str) -> str:
    """ Turns the file path of an image into the correct label for that image. """
    matched = REGEX_MACHINE.search(path).group(0)
    disease = matched[1:-1]
    return disease


def get_pixel_series(path: str) -> pd.Series:
    """ Takes the input file path and returns a Pandas Series object containing all pixel values. """
    image = Image.open(path)
    red, _, _ = image.split()
    flat = np.array(red).flatten()
    return pd.Series(flat)


if __name__ == "__main__":
    path_string = "split/ptb-images-2-cropped/train/ValvularHeartDisease/patient106_s0030_re_1_ValvularHeartDisease.jpg"
    fnames, targets = get_filenames_and_labels()
    print(len(fnames))
