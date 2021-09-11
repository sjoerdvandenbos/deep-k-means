import tensorflow as tf
import numpy as np
import pandas as pd
import skimage.measure

from ptb_img_utils import get_filenames_and_labels
from utils import read_list


# Note that data here is only a list with filenames, not the actual images.
filenames, diseases = get_filenames_and_labels("split/ptb-matrices")

print("loading matrix data", flush=True)

data = []
for f in filenames:
    matrix = pd.read_csv(f, sep=",", header=None, nrows=15, low_memory=False, dtype=np.float64, engine="c").to_numpy()
    reduced = skimage.measure.block_reduce(matrix, (1, 7), np.max, cval=-3)
    data.append(reduced.flatten())
data = np.asarray(data)

print("matrix data loaded", flush=True)

n_clusters = 7
disease_mapping = {
    "BundleBranchBlock": 0,
    "Cardiomyopathy": 1,
    "Dysrhythmia": 2,
    "HealthyControl": 3,
    "MyocardialInfarction": 4,
    "Myocarditis": 5,
    "ValvularHeartDisease": 6
}
target = np.array([disease_mapping[d] for d in diseases])

# Get the split between training/test set and validation set
test_indices = read_list("split/ptb-matrices/test")
train_indices = read_list("split/ptb-matrices/validation")

n_samples = len(train_indices)

# Auto-encoder architecture
input_size = 15 * 143
hidden_1_size = 500
hidden_2_size = 500
hidden_3_size = 2000
embedding_size = n_clusters
dimensions = [hidden_1_size, hidden_2_size, hidden_3_size, embedding_size, # Encoder layer dimensions
              hidden_3_size, hidden_2_size, hidden_1_size, input_size] # Decoder layer dimensions
activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, None, # Encoder layer activations
               tf.nn.relu, tf.nn.relu, tf.nn.relu, None] # Decoder layer activations
names = ['enc_hidden_1', 'enc_hidden_2', 'enc_hidden_3', 'embedding', # Encoder layer names
         'dec_hidden_1', 'dec_hidden_2', 'dec_hidden_3', 'output'] # Decoder layer names