import glob

import pandas as pd
import numpy as np


# Training and test data can be divided by sorting
all_files = sorted(glob.glob("split/ptb-matrices/*/*/*"))
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

training.to_csv("split/ptb-matrices/validation", index=False, header=False)
testing.to_csv("split/ptb-matrices/test", index=False, header=False)
