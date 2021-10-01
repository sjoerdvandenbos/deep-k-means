import torch
from PIL import Image
import numpy as np
import skimage.measure
from matplotlib import cm
from torch.utils.data import Dataset

from glob import glob
from re import compile
from pathlib import Path

from utils import read_list

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
    reduced = skimage.measure.block_reduce(cropped, (7, 7), np.min)
    return reduced.flatten().astype(np.uint8)


def save_imgs_to_npy(path):
    data_path = Path.cwd() / path
    print("Loading data into memory...", flush=True)
    filenames, diseases = get_filenames_and_labels(data_path)
    data = np.asarray([get_image_tensor(f) for f in filenames], dtype=np.uint8)
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
        return Image.fromarray(np.uint8(cm.gist_earth(transformed) * 255))
    else:
        return Image.fromarray(np.uint8(cm.gist_earth(numpy_right_shape) * 255))


class PTBImgSet(Dataset):

    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.data = torch.as_tensor(source)
        self.target = torch.as_tensor(target)

    def __getitem__(self, index):
        transformed = (self.data[index] / 255).float()
        return index, transformed, self.target[index]

    def __len__(self):
        return self.data.shape[0]


def get_train_and_test_sets(path: Path):
    all_data = np.load(path / "compacted_data.npy")
    train_indices = read_list(path / "train")
    test_indices = read_list(path / "validation")
    train_data = all_data[train_indices]
    test_data = all_data[test_indices]
    all_targets = np.load(path / "compacted_target.npy")
    train_target = np.fromiter((DISEASE_MAPPING[d] for d in all_targets[train_indices]),
                               dtype=np.uint8)
    test_target = np.fromiter((DISEASE_MAPPING[d] for d in all_targets[test_indices]),
                              dtype=np.uint8)
    train_set = PTBImgSet(train_data, train_target)
    test_set = PTBImgSet(test_data, test_target)
    return train_set, test_set


def autoencoder_loss(x, y):
    return torch.square(x - y).sum(dim=0).mean()


def kmeans_distance(x, y):
    return torch.square(x - y).sum(dim=1)
