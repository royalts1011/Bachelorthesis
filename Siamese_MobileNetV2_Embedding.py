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


# %%
class Config():
    DEVICE = get_device()
    DATASET_DIR = '../dataset/'
    AUTH_DATASET_DIR = '../auth_dataset/unknown-auth/'
    MODEL_DIR = './models/model_1.pt'
    RESIZE_SMALL = False
    DATABASE_FOLDER = './embeddings/'


# %%
model = torch.load(Config.MODEL_DIR, map_location=torch.device(Config.DEVICE))
transformation = td.transforms_valid_and_test( td.get_resize(Config.RESIZE_SMALL) )
#model.eval()


# %%
def pipeline(input_, preprocess):
    input_ = input_.convert("L")
    input_ = preprocess(input_)
    input_ = input_.reshape(-1, td.get_resize(Config.RESIZE_SMALL)[0], td.get_resize(Config.RESIZE_SMALL)[1], 1)
    input_ = input_.permute(3, 0, 1, 2)

    if cuda.is_available():
        return input_.type('torch.cuda.FloatTensor')
    else:
        return input_.type('torch.FloatTensor')


# %%
# img = Image.open(Config.DATASET_DIR + 'janna_qua/janna_qua003.png')
# new_embedding = [model(Variable(pipeline(img,transformation))).cpu()]
# test = np.array(new_embedding)

# np.save('test.npy', test)


# %%
# loaded_test = np.load('test.npy', allow_pickle=True)

# print(loaded_test)
# print(new_embedding)

# value = F.pairwise_distance(loaded_test[0],new_embedding[0]).item()
# print(value)


# %%
result_value = []
result_label = []

img = Image.open(Config.DATASET_DIR + 'janna_qua/janna_qua003.png')
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


# %%


