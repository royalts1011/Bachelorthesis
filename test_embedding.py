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
import acquire_ear_dataset as a
import ds_ear_siamese as des
import transforms_data as td

from PIL import Image
import glob
import os


# %%
class Config():
    DEVICE = get_device()
    DATASET_DIR = '../dataset/'
    MODEL_DIR = './models/model_MN_color.pt'
    RESIZE_SMALL = False
    DATABASE_FOLDER = './embeddings/'


# %%
model = torch.load(Config.MODEL_DIR, map_location=torch.device(Config.DEVICE))
preprocess = td.transforms_siamese_verification( td.get_resize(Config.RESIZE_SMALL) )


# %%
def image_pipeline(input_, preprocess):
    #input_ = input_.convert("L")
    input_ = preprocess(input_)
    input_ = input_.reshape(-1, td.get_resize(Config.RESIZE_SMALL)[0], td.get_resize(Config.RESIZE_SMALL)[1], 1)
    input_ = input_.permute(3, 0, 1, 2)

    if cuda.is_available():
        return input_.type('torch.cuda.FloatTensor')
    else:
        return input_.type('torch.FloatTensor')


# %%
a.capture_ear_images(amount_pic=3, pic_per_stage=3, is_authentification=True)


# %%
result_value = []
result_label = []

img = Image.open('/Users/falcolentzsch/Develope/Bachelorthesis/auth_dataset/unknown-auth/unknown002.png')
new_embedding = model(Variable(image_pipeline(img,preprocess))).cpu()

for label in os.listdir(Config.DATABASE_FOLDER):
    loaded_embeddings = np.load(Config.DATABASE_FOLDER+label, allow_pickle=True)
    tmp = []    
    for embedding in loaded_embeddings:
        dis = F.pairwise_distance(embedding,new_embedding)
        tmp.append(dis.item())
    result_value.append((min(tmp)))
    result_label.append(label)


# %%
result_value, result_label = zip(*sorted(zip(result_value, result_label)))
result_value = result_value[:5]
result_label = result_label[:5]

for idx, val in enumerate(result_label):
    print(str(idx+1) + ' : ' + ' ' + val + ' : ' + ' ' + str(result_value[idx]))


# %%
shutil.rmtree('../auth_dataset/unknown-auth')


