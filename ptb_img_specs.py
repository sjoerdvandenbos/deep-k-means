import tensorflow as tf
import numpy as np
from pathlib import Path

from ptb_img_utils import DISEASE_MAPPING
from utils import read_list


data_path = Path.cwd() / "split" / "ptb-images-2-cropped" / "3k_per_disease"
print("Loading data...")
data = np.load(data_path / "compacted_data.npy").astype(np.uint8).reshape((-1, 120576)) / 255
diseases = np.load(data_path / "compacted_target.npy").astype(np.str).flatten()
print(data.shape)
print("Done loading data")
n_clusters = 7
target = np.asarray([DISEASE_MAPPING[d] for d in diseases])

# Get the split between training/test set and validation set
test_indices = read_list(data_path / "test")
train_indices = read_list(data_path / "validation")

n_samples = train_indices.shape[0]

# Auto-encoder architecture
input_size = 120576
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