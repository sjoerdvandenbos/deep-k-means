from pathlib import Path

import numpy as np

root = Path(__file__).parent
path = root / "split" / "ptb-concatted-leads" / "compacted_data.npy"

data = np.load(path)

correct_format = np.expand_dims(data, axis=1)

np.save(path, correct_format)
