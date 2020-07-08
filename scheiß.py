# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
sys.path.append('../..')
import numpy as np
import matplotlib.pyplot as plt

# PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
from torch.autograd import Variable
from torchvision.utils import make_grid
from torchvision.models.mobilenet import mobilenet_v2

# DLBio and own scripts
from DLBio.pytorch_helpers import get_device
import ds_ear_siamese
import transforms_data as td
from helpers import cuda_conv
import metrics as M
from siamese_network_train import Training
from ContrastiveLossFunction import ContrastiveLoss
from NN_Siamese import SiameseNetwork

from PIL import Image
import glob


# %%
class Config():
    DEVICE = get_device()
    DATASET_DIR = '../dataset/'
    MODEL_DIR = './models/model_MN_allunfreezed.pt'
    RESIZE_SMALL = False
    DATABASE_FOLDER = './embeddings/'


# %%
model = torch.load(Config.MODEL_DIR)
model.to(Config.DEVICE)


# %%
test = np.load(Config.DATABASE_FOLDER+'alexander_bec.npy', allow_pickle=True)

a = 1

# %%


