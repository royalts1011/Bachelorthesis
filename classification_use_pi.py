# %%
import sys
sys.path.append('../..')
import torch
import numpy as np
import transforms_data as td
from PIL import Image
import glob
from torch import cuda
import acquire_ear_dataset as a
import os
import shutil
from DLBio.pytorch_helpers import get_device
import time
import RPi.GPIO as GPIO



CATEGORIES = ["mila_wol", "falco_len", "jesse_kru", "konrad_von", "nils_loo", "johannes_boe", "johannes_wie", "sarah_feh", "janna_qua", "tim_moe"]
CATEGORIES.sort()
AUTHORIZED = ["falco_len","konrad_von"]
RESIZE_Y = 150
RESIZE_X = 100
DATA_TEST_FOLDER = "../auth_dataset/unknown-auth/*png"
DEVICE = get_device()

model = torch.load('./class_sample/model.pt', DEVICE)


# %%
# Bilder aufnehmen
GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.OUT)
GPIO.output(4, True)
a.capture_ear_images(amount_pic=10, pic_per_stage=10, is_authentification=True)
GPIO.output(4, False)


# %%
image_array = []
files = glob.glob (DATA_TEST_FOLDER)
files.sort()
# declare function of transformation
preprocess = td.transforms_valid_and_test((RESIZE_Y, RESIZE_X),[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

for f in files:
    image = Image.open(f)
    image_transformed = preprocess(image)
    image_transformed = image_transformed.reshape(-1, RESIZE_Y, RESIZE_X, 1)
    image_transformed = image_transformed.permute(3, 0, 1, 2)
    if cuda.is_available():
        image_array.append(image_transformed.type('torch.cuda.FloatTensor'))
    else:
        image_array.append(image_transformed.type('torch.FloatTensor'))


# %%
all_classes = []
summ_pred = np.zeros(1)
for i in image_array:
	with torch.no_grad():
		pred = model(i)
		pred = torch.softmax(pred, 1)
		pred = pred.cpu().numpy()
		summ_pred = summ_pred + pred

	classes = np.argmax(pred, 1)
	all_classes.append(classes[0])

	pred = np.append(pred, classes)
	pred = np.append(pred, CATEGORIES[classes[0]])	
	print(pred, "\n")
print(all_classes)
print(summ_pred)


NUMBER_AUTHORIZED = int(.3*len(image_array))
authentification_dict = {CATEGORIES[i]:all_classes.count(i) for i in all_classes}
print(authentification_dict) 
acces = False
for a in authentification_dict:
    if a in AUTHORIZED and summ_pred[0][CATEGORIES.index(a)]>= NUMBER_AUTHORIZED:
        print("Access granted! Welcome "  + a + "!")
        GPIO.setup(17, GPIO.OUT)
        GPIO.output(17, True)
        time.sleep(10)
        GPIO.output(17, False)
        acces = True
        break

if not acces :
    print("Acces Denied")
    GPIO.setup(27, GPIO.OUT)
    GPIO.output(27, True)
    time.sleep(10)
    GPIO.output(27, False)


# %%
shutil.rmtree('../auth_dataset/unknown-auth')
