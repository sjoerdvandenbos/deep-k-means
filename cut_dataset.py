import glob
import os
import numpy as np


def reduce_size(zipped, number_every_disease):
    """
    @param zipped: A zipped list with tuples of the same disease dirs from image and csv respectively.
    @param number_every_disease: number of samples to retain in every disease folder.
    """
    for image_dir, csv_dir in zipped:
        images = sorted(glob.glob(f"{image_dir}/*.jpg"))
        csvs = sorted(glob.glob(f"{csv_dir}/*.csv"))

        rand_indices = np.arange(0, len(csvs))
        np.random.shuffle(rand_indices)
        indices_to_delete = rand_indices[number_every_disease:]

        for i in indices_to_delete:
            os.remove(images[i])
            os.remove(csvs[i])

        size_image_dir = len(glob.glob(f"{image_dir}/*.jpg"))
        size_csv_dir = len(glob.glob(f"{csv_dir}/*.csv"))
        print(f"{size_image_dir} files in the img dir, {size_csv_dir} files in the csv dir.")


if __name__ == "__main__":
    train_image_dirs = sorted(glob.glob("split/ptb-images-2-cropped/train/*"))
    train_csv_dirs = sorted(glob.glob("split/ptb-matrices/train/*"))
    test_image_dirs = sorted(glob.glob("split/ptb-images-2-cropped/val/*"))
    test_csv_dirs = sorted(glob.glob("split/ptb-matrices/val/*"))

    reduce_size(zip(train_image_dirs, train_csv_dirs), 1000)
    reduce_size(zip(test_image_dirs, test_csv_dirs), 256)
