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

# Pin imports
from gpiozero import LED
import RPi.GPIO as GPIO
from Adafruit_CharLCD import Adafruit_CharLCD
import shutil
import os
from time import sleep


# %%
# instantiate lcd and specify pins
lcd = Adafruit_CharLCD(rs=26, en=19,
                       d4=13, d5=6, d6=5, d7=11,
                       cols=16, lines=2)
# initiate LEDs
led_yellow = LED(4)
led_green = LED(17)
led_red = LED(27)


# %%
class Config():
    DEVICE = get_device()
    DATASET_DIR = '../dataset/'
    AUTH_DATASET_DIR = '../auth_dataset/unknown-auth/'
    MODEL_DIR = './models/ve_g_margin_2,0.pt'
    is_small_resize = False
    DATABASE_FOLDER = './embeddings/radius_2.0/'
    THRESHOLD_VAL = 0.9
    THRESHOLD = 2.0
    a = 0.1
    AUTHORIZED = ["falco_len.npy","konrad_von.npy"]


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
# Verification
try:
    lcd.blink(False)
    

    # LCD output
    lcd.clear()
    lcd.message('Ready to take\nyour ear images')

    # Bilder aufnehmen
    led_yellow.blink(on_time=0.5,off_time=0.25)

    a.capture_ear_images(amount_pic=4, pic_per_stage=4, is_authentification=True)
    # Die ersten Bilder entfernen, da häufig verschwommen
    os.remove('../auth_dataset/unknown-auth/unknown001.png')
    os.remove('../auth_dataset/unknown-auth/unknown002.png')
    os.remove('../auth_dataset/unknown-auth/unknown003.png')

    led_yellow.off()
    
    # LCD output
    lcd.clear()
    lcd.message('Verification\nin progress...')

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


   # access = False

    if (result_value[0] + Config.a) < result_value[1] and result_value[0] <= Config.THRESHOLD_VAL and result_label[0] in Config.AUTHORIZED:
        # LCD output
        lcd.clear()
        entry_string = 'Hi ' + result_label[0]
        lcd.message('Access granted\n'+ entry_string)

        print("Access granted! Welcome "  + result_label[0] + "!")

        led_green.on()
        sleep(10)
        led_green.off()


    #%%
    if result_value[0] > Config.THRESHOLD_VAL and result_value[0] < Config.THRESHOLD:
        lcd.clear()
        lcd.message('Not found -\nNo entry.')
        
        print('Cant find authorized Person in Database. Pls try again')

        led_red.on()
        sleep(10)
        led_red.off() 

    if (result_value[0] + Config.a) >= result_value[1]:
        lcd.clear()
        lcd.message('Access denied -\nNo entry.')
                
        print('Verification not clear enough. Access denied. Please try again.')
        led_red.on()
        sleep(10)
        led_red.off() 



finally:
    # clear outputs
    lcd.clear()

    # %%
    shutil.rmtree('../auth_dataset/unknown-auth')

