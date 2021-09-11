import glob

import pandas as pd
import numpy as np


def split_indices(dataset: str) -> None:
    # Training and test data can be divided by sorting
    all_files = sorted(glob.glob(f"split/{dataset}/*/*/*"))
    first_train, first_test = None, None
    for i, f in enumerate(all_files):
        if first_train is None and "train" in f:
            first_train = i
        # Test folder is called val
        if first_test is None and "val" in f:
            first_test = i
    print(f"first train index: {first_train}")
    print(f"first test index: {first_test}")
    indices = np.arange(0, len(all_files))
    training = pd.DataFrame(indices[:first_test])
    testing = pd.DataFrame(indices[first_test:])
    training.to_csv(f"split/{dataset}/validation", index=False, header=False)
    testing.to_csv(f"split/{dataset}/test", index=False, header=False)


if __name__ == "__main__":
    split_indices("ptb-images-2-cropped")
    split_indices("ptb-matrices")
