activate = '/home/pi/.virtualenvs/Bachelorthesis/bin/activate_this.py'
exec(open(activate).read(),{'__file__': activate})


import sys
sys.path.append('../..')
sys.path.append('/home/pi/Documents/Bachelorarbeit/')
import numpy as np
from PIL import Image
import glob
import os
import shutil
from time import sleep

# Pytorch
import torch
from torch import cuda

# DLBio and own scripts
import transforms_data as td
import helpers as hp
import ds_ear
import acquire_ear_dataset as a
from DLBio.pytorch_helpers import get_device

# Pin imports
from gpiozero import LED
import RPi.GPIO as GPIO
from Adafruit_CharLCD import Adafruit_CharLCD



# DATASET_DIR = '../dataset_low_res/'
# CATEGORIES = ds_ear.get_dataset(DATASET_DIR, transform_mode='size_only').classes
# #CATEGORIES = ["mila_wol", "falco_len", "jesse_kru", "konrad_von", "nils_loo", "johannes_boe", "johannes_wie", "sarah_feh", "janna_qua", "tim_moe"]
# # CATEGORIES = ["falco_len", "nils_loo", "alissa_buh", "gregor_spi"]
# CATEGORIES.sort()
# AUTHORIZED = ["falco_len"]
# DATA_TEST_DIR = "../auth_dataset/unknown-auth/*png"
# DEVICE = get_device()

class Config():
    DATASET_DIR = '../dataset/'
    dset = ds_ear.get_dataset(DATASET_DIR, transform_mode='size_only')
    CATEGORIES = dset.classes
    # CATEGORIES = ["mila_wol", "falco_len", "jesse_kru", "konrad_von", "nils_loo", "johannes_boe", "johannes_wie", "sarah_feh", "janna_qua", "tim_moe"]
    CATEGORIES.sort()
    AUTHORIZED = ["falco_len","konrad_von"]
    DATA_TEST_DIR = "../auth_dataset/unknown-auth/*png"
    DEVICE = get_device()
    RESIZE_SMALL = False



model = torch.load('./models/cl36_c_9091.pt', Config.DEVICE)

# instantiate lcd and specify pins
lcd = Adafruit_CharLCD(rs=26, en=19,
                       d4=13, d5=6, d6=5, d7=11,
                       cols=16, lines=2)
# initiate LEDs
led_yellow = LED(4)
led_green = LED(17)
led_red = LED(27)

try:

    lcd.blink(False)
    lcd.clear()


    # LCD output
    lcd.message('Ready to take\nyour ear images')

    # %%
    # Bilder aufnehmen
    led_yellow.blink(on_time=0.5,off_time=0.25)
    a.capture_ear_images(amount_pic=10, pic_per_stage=10, is_authentification=True)
    led_yellow.off()

    # LCD output
    lcd.clear()
    lcd.message('Authentication\nin progress...')

    # %%
    image_array = []
    files = glob.glob (Config.DATA_TEST_DIR)
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
        
        image_array.append( hp.type_conversion(image_transformed) )


    # %%
    all_classes = []
    summ_pred = np.zeros(1)
    for i in image_array:
        with torch.no_grad():
            pred = model(i)
            pred = torch.softmax(pred, 1)
            pred = pred.cpu().numpy()
            summ_pred = summ_pred + pred
            
        hp.print_predictions(Config.CATEGORIES,pred[0])
        class_ = np.argmax(pred, 1)
        all_classes.append(class_[0])

        print('Highest value: ', Config.CATEGORIES[class_[0]], '\n')

    unique = list(set( [Config.CATEGORIES[c] for c in all_classes] ))
    print('\n'*2, '#'*40)
    print('Accumulated predictions:')
    hp.print_predictions(
            unique,
            [summ_pred[0][Config.dset.class_to_idx[name]] for name in unique]
            )


    num_authorized = int(.3*len(image_array))
    authentification_dict = {Config.CATEGORIES[i]:all_classes.count(i) for i in all_classes}
    print('\nFrequency of prediction:')
    fmt = '{:<20} {:<4}'
    for key, value in authentification_dict.items():
        print(fmt.format(key, value))

    access = False
    for a in authentification_dict:
        if a in Config.AUTHORIZED and summ_pred[0][Config.CATEGORIES.index(a)]>= num_authorized:
            
            # LCD output
            lcd.clear()
            entry_string = 'Hi ' + a
            lcd.message('Access granted\n'+ entry_string)
            
            # scroll through whole text until its gone
            if(len(entry_string)>16):            
                sleep(2)
                # scroll text on display
                for x in range(entry_string):
                    lcd.move_left()
                    sleep(.5)
                lcd.home()
                lcd.message('Access granted -\nWelcome!')
            
            print("\n\t~~~ Access granted! Welcome "  + a + "! ~~~")
            led_green.on()
            sleep(10)
            led_green.off()
            access = True
            break

    if not access :
        # LCD output
        lcd.clear()
        lcd.message('Access denied -\nNo entry.')

        print("\n\t~~~ Access denied ~~~")
        led_red.on()
        sleep(10)
        led_red.off()

finally:
    # clear outputs
    lcd.clear()

    # %%
    shutil.rmtree('../auth_dataset/unknown-auth')