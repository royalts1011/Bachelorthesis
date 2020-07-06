# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
sys.path.append('../..')
import numpy as np
from PIL import Image
import glob
import os
import shutil

# Pytorch
import torch
from torch import cuda

# DLBio and own scripts
import transforms_data as td
import ds_ear
import helpers
import acquire_ear_dataset as a
from DLBio.pytorch_helpers import get_device


class Config():
    DATASET_DIR = '../dataset_low_res/'
    CATEGORIES = ds_ear.get_dataset(DATASET_DIR, transform_mode='size_only').classes
    # CATEGORIES = ["mila_wol", "falco_len", "jesse_kru", "konrad_von", "nils_loo", "johannes_boe", "johannes_wie", "sarah_feh", "janna_qua", "tim_moe"]
    CATEGORIES.sort()
    AUTHORIZED = ["falco_len","konrad_von"]
    DATA_TEST_DIR = "../auth_dataset/unknown-auth/*png"
    RESIZE_SMALL = True
    DEVICE = get_device()

model = torch.load('./class_sample/model.pt', Config.DEVICE)


# %%
# Bilder aufnehmen
a.capture_ear_images(amount_pic=10, pic_per_stage=10, is_authentification=True)


# %%
image_array = []
files = glob.glob(Config.DATA_TEST_DIR)
files.sort()
# declare function of transformation
preprocess = td.transforms_valid_and_test( td.get_resize(small=Config.RESIZE_SMALL) )

for f in files:
    image = Image.open(f)
    image_transformed = preprocess(image)
    image_transformed = image_transformed.reshape(
                            -1,
                            td.get_resize(small=Config.RESIZE_SMALL)[0],
                            td.get_resize(small=Config.RESIZE_SMALL)[1],
                            1
                            )
    image_transformed = image_transformed.permute(3, 0, 1, 2)

    image_array.append( helpers.type_conversion(image_transformed) )


# %%
all_classes = []
summ_pred = np.zeros(1)
print('\nClass predictions:')
for i in image_array:
    with torch.no_grad():
        pred = model(i)
        pred = torch.softmax(pred, 1)
        pred = pred.cpu().numpy()
        summ_pred = summ_pred + pred
    # Print probability of class
    helpers.print_predictions(Config.CATEGORIES,pred[0])
    class_ = np.argmax(pred, 1)
    all_classes.append(class_[0])

    print('Highest value: ', Config.CATEGORIES[class_[0]], '\n')
    # pred = np.append(pred, class_)
    # pred = np.append(pred, Config.CATEGORIES[class_[0]])	
	# print(pred)

print('\n'*2, '#'*40)
print('Accumulated predictions:')
helpers.print_predictions(
        [Config.CATEGORIES[c] for c in all_classes],
        list(summ_pred[0])
        )
# print(all_classes)
# print(summ_pred)


# %%
num_authorized = int(.3*len(image_array))
authentification_dict = {Config.CATEGORIES[i]:all_classes.count(i) for i in all_classes}
print('\nFrequency of prediction:')
fmt = '{:<20} {:<4}'
for key, value in authentification_dict.items():
    print(fmt.format(key, value))

for a in authentification_dict:
    if a in Config.AUTHORIZED and summ_pred[0][Config.CATEGORIES.index(a)]>= num_authorized:
        print("\n\t~~~ Access granted! Welcome "  + a + "! ~~~")
        break
    else:
        print("\n\t~~~ Access denied ~~~")
        break


# %%
shutil.rmtree('../auth_dataset/unknown-auth')


# %%


