# %%
# math and system imports
import sys
sys.path.append('../..')
import numpy as np
from PIL import Image
import glob
import shutil
import os
from os.path import join
import time

# PyTorch imports
import torch
from torch import cuda
from torch.autograd import Variable
import torch.nn.functional as F

# own scripts
import acquire_ear_dataset as a
import ds_ear_siamese as des
import transforms_data as td
import helpers
from DLBio.pytorch_helpers import get_device


# %%
# central variable hub
class Config():
    NN_SIAMESE = False
    AUTHORIZED = ["falco_len","konrad_von"]
    DEVICE = get_device()

    DATASET_DIR = '../dataset_low_res/'
    VERIFICATION_DIR = '../auth_dataset/unknown-auth'
    MODEL_DIR = './models/ve_g_9997.pt'

    is_small_resize = True

    TRESHOLD = 3.0
    TRESHOLD_VER = 0.8
    a = 0

model = torch.load(Config.MODEL_DIR, Config.DEVICE) 


# %%
#Output for NN  
if Config.NN_SIAMESE == True:
    def generate_output( img_in, img_in2 ):
        if cuda.is_available():
            return model(Variable(img_in).cuda(), Variable(img_in2).cuda())
        else:
            return model(Variable(img_in), Variable(img_in2))
else:
# Output for Mobilenet
    def generate_output( img_in ):
        if cuda.is_available():
            return model(Variable(img_in).cuda())
        else:
            return model(Variable(img_in))    


# %%
a.capture_ear_images(amount_pic=10, pic_per_stage=10, is_authentification=True)


# %%
# Image processing
# input is the filename and the transforamtion function
def image_pipeline(input_, preprocess):
    input_ = Image.open(input_)
    input_ = input_.convert("L")
    input_ = preprocess(input_)
    input_ = input_.reshape(-1, td.get_resize(Config.is_small_resize)[0], td.get_resize(Config.is_small_resize)[1], 1)
    input_ = input_.permute(3, 0, 1, 2)
    
    if cuda.is_available():
        return input_.type('torch.cuda.FloatTensor')
    else:
        return input_.type('torch.FloatTensor')


# %%
# Setting up the dataset
def get_triplets(dataset_path, user_name, verif_dataset):
    dataset_classes = helpers.rm_DSStore( os.listdir( dataset_path ) )
    NUM_CLASSES = len(dataset_classes)
    user_imgs = helpers.rm_DSStore( os.listdir(join(dataset_path, user_name)) )
    
    preprocess = td.get_transform('valid_and_test', Config.is_small_resize)

    # Triplets list will contain anchor(A), positive(P) and negative(N) triplets.
    triplets = []

    # creating anchor, positive, negative triplets (amount equals taken images)
    for img_dir in helpers.rm_DSStore( os.listdir(verif_dataset) ):
        anchor_dir = join(verif_dataset, img_dir)
        anchor = image_pipeline(anchor_dir, preprocess)

        # choose random image from the alleged user and pass the preprocess strategy
        positive = image_pipeline( join(dataset_path, user_name, user_imgs[np.random.randint(0, len(user_imgs))]), preprocess )
        
        # find a class different from anchor. if same as anchor try again
        neg_class = dataset_classes[np.random.randint(NUM_CLASSES)]
        while neg_class == anchor_dir: 
            neg_class = dataset_classes[np.random.randint(NUM_CLASSES)]
        # list of image file names
        neg_class_imgs = helpers.rm_DSStore( os.listdir(join(dataset_path, neg_class)) )
        # choose random image from the negative and pass the preprocess strategy
        negative = image_pipeline( join(dataset_path, neg_class, neg_class_imgs[np.random.randint(0, len(neg_class_imgs))]), preprocess )

        # append triplet
        triplets.append([anchor, positive, negative])

    return triplets


# %%
# pers_to_ver = input('\n Enter name end press <return> ==>  ')
pers_to_ver = helpers.choose_folder(Config.DATASET_DIR)

start_time = time.time()

triplet_list = get_triplets(dataset_path=Config.DATASET_DIR,
                            user_name=pers_to_ver,
                            verif_dataset=Config.VERIFICATION_DIR)


# %%
verification_counter = 0
#Scaling factor
for t in triplet_list:
    if Config.NN_SIAMESE == True:
        match_out1, match_out2 = generate_output(t[0], t[1])
        #non_match_out1, non_match_out2 = generate_output(t[0], t[2])
    else:
        match_out1 = generate_output(t[0])
        match_out2 = generate_output(t[1])                 
        #non_match_out1 = generate_output(t[0])
        #non_match_out2 = generate_output(t[2])

    euclidean_distance_pp = F.pairwise_distance(match_out1, match_out2)
    #euclidean_distance_pn = F.pairwise_distance(non_match_out1, non_match_out2)
    #if(euclidean_distance_pp >= Config.TRESHOLD): continue
    
    #if(euclidean_distance_pp + Config.a < euclidean_distance_pn): verification_counter += 1
    if(euclidean_distance_pp + Config.a < Config.TRESHOLD_VER): verification_counter += 1

    # format variables
    fmt_id = '{:<12}'
    fmt_eucl = '{:<.3f}'
    print(fmt_id.format('pos-pos: '), fmt_eucl.format( euclidean_distance_pp.item()) )
    #print(fmt_id.format('pos-neg: '),fmt_eucl.format( euclidean_distance_pn.item()) )
    print(fmt_id.format('Acc. count: '), '{:>.0f}'.format(verification_counter), '\n')


# %%
NUMBER_AUTHORIZED = int(.8*len(triplet_list))
if  pers_to_ver in Config.AUTHORIZED and verification_counter >= NUMBER_AUTHORIZED:
    print("Access granted! Welcome "  + pers_to_ver + "!")
else:
    print("Access denied")


# %%
shutil.rmtree('../auth_dataset/unknown-auth')
print(time.time() - start_time)
