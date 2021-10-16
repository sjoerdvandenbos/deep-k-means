#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

import tensorflow as tf
from utils import read_list
from sklearn.datasets import fetch_openml
from ptb_img_utils import PTBImgSet
import numpy as np

# Fetch the dataset
download = fetch_openml("mnist_784", version=1, cache=True)
print("Dataset MNIST loaded...")
data = download.data.to_numpy()
target = download.target.to_numpy().astype(np.int64)
img_height = 28
img_width = 28
n_samples = data.shape[0] # Number of samples in the dataset
n_clusters = 10 # Number of clusters to obtain
data = data.reshape(n_samples, 1, img_height, img_width)

# Get the split between training/test set and validation set
train_indices = read_list("split/mnist/train")
test_indices = read_list("split/mnist/validation")

trainset = PTBImgSet(data[train_indices], target[train_indices])
testset = PTBImgSet(data[test_indices], target[test_indices])

# Auto-encoder architecture
input_size = img_height * img_width
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