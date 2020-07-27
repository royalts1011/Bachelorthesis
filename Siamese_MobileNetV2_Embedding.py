# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
sys.path.append('../..')
import numpy as np
# PyTorch
import torch
import torch.nn.functional as F
from torch import cuda
from torch.autograd import Variable
from torchvision.models.mobilenet import mobilenet_v2

# DLBio and own scripts
from DLBio.pytorch_helpers import get_device
import transforms_data as td
from helpers import cuda_conv
import metrics as M
import acquire_ear_dataset as a

from PIL import Image
import glob
import os
import shutil


# %%
class Config():
    DEVICE = get_device()
    DATASET_DIR = '../dataset/'
    AUTH_DATASET_DIR = '../auth_dataset/unknown-auth/'
    MODEL_DIR = './models/ve_g_9997.pt'
    is_small_resize = False
    DATABASE_FOLDER = './embeddings/'
    THRESHOLD_VAL = 1.0
    THRESHOLD = 2.0
    a = 0.1


# %%
model = torch.load(Config.MODEL_DIR, map_location=torch.device(Config.DEVICE))
transformation = td.get_transform('siamese_valid_and_test', Config.is_small_resize)
#model.eval()


# %%
def pipeline(input_, preprocess):
    input_ = input_.convert("L")
    input_ = preprocess(input_)
    input_ = input_.reshape(-1, td.get_resize(Config.is_small_resize)[0], td.get_resize(Config.is_small_resize)[1], 1)
    input_ = input_.permute(3, 0, 1, 2)

    if cuda.is_available():
        return input_.type('torch.cuda.FloatTensor')
    else:
        return input_.type('torch.FloatTensor')


# %%
a.capture_ear_images(amount_pic=4, pic_per_stage=4, is_authentification=True)
# Die ersten Bilder entfernen, da häufig verschwommen
os.remove('../auth_dataset/unknown-auth/unknown001.png')
os.remove('../auth_dataset/unknown-auth/unknown002.png')
os.remove('../auth_dataset/unknown-auth/unknown003.png')

# %%
result_value = []
result_label = []

img = Image.open(Config.AUTH_DATASET_DIR + 'unknown004.png')
new_embedding = model(Variable(pipeline(img,transformation))).cpu()

for label in os.listdir(Config.DATABASE_FOLDER):
    loaded_embedding = np.load(Config.DATABASE_FOLDER+label, allow_pickle=True)
    tmp = []    
    for embedding in loaded_embedding:
        dis = F.pairwise_distance(embedding,new_embedding)
        tmp.append(dis.item())
    result_value.append((min(tmp)))
    result_label.append(label)


# %%
result_value, result_label = zip(*sorted(zip(result_value, result_label)))
result_value = result_value[:10]
result_label = result_label[:10]

for idx, val in enumerate(result_label):
    print(str(idx+1) + ' : ' + ' ' + val + ' : ' + ' ' + str(result_value[idx]))




#%%
if result_value[0] > Config.THRESHOLD_VAL and result_value[0] < Config.THRESHOLD:
    print('Cant find authorized Person in Database. Pls try again')

if (result_value[0] + Config.a) >= result_value[1]:
    print('Verification not clear enough. Access denied. Please try again.')

if (result_value[0] + Config.a) < result_value[1] and result_value[0] <= Config.THRESHOLD_VAL:
    print("Access granted! Welcome "  + result_label[0] + "!")


# %%
shutil.rmtree('../auth_dataset/unknown-auth')

