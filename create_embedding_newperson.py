# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
sys.path.append('../..')
from os.path import join, exists
import numpy as np
from PIL import Image
import glob
# PyTorch
import torch
from torch import cuda
from torch.autograd import Variable
# DLBio and own scripts
from DLBio.pytorch_helpers import get_device
import transforms_data as td
import helpers as hp


# %%
class Config():
    DEVICE = get_device()
    DATASET_DIR = '../dataset/'
    MODEL_DIR = './models/ve_g_margin_2,0.pt'
    is_small_resize = False
    DATABASE_FOLDER = './embeddings/radius_2.0'


# %%
new_person = hp.choose_folder(dataset_path=Config.DATASET_DIR)
check = exists(join(Config.DATABASE_FOLDER, new_person+'.npy'))


# %%
if check: print('The embedding of that person might already exist. Please check the folder first!')

print('You chose ', new_person, ' to be processed.', '\n ABORT NOW, OR:')
input('Press any key to continue.')


# %%
model = torch.load(Config.MODEL_DIR, map_location=torch.device(Config.DEVICE))
transformation = td.get_transform('siamese_valid_and_test', Config.is_small_resize)


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
embeddings = []
image_list = []
for filename in glob.glob( join(Config.DATASET_DIR, new_person, '*') ):
    img = Image.open(filename)
    img_processed = pipeline(img,transformation)
    image_list.append(img_processed)
    
embeddings = np.array([model(Variable(i)).cpu() for i in image_list])
    
np.save( join(Config.DATABASE_FOLDER,new_person+'.npy'), embeddings)    


