import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

LOG_DIR = 'minimalsample'
NAME_TO_VISUALISE_VARIABLE = "ears"
TO_EMBED_COUNT = 500


path_for_mnist_sprites =  os.path.join(LOG_DIR,'ear.png')
path_for_mnist_metadata =  os.path.join(LOG_DIR,'metadata.tsv')


ears = input_data.read_data_sets('../dataset/', one_hot=False)
batch_xs, batch_ys = ears.train.next_batch(TO_EMBED_COUNT)