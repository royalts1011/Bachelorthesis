activate = '/home/pi/.virtualenvs/Bachelorthesis/bin/activate_this.py'
exec(open(activate).read(),{'__file__': activate})

# %%
import sys
sys.path.append('../..')
sys.path.append('/home/pi/Documents/Bachelorarbeit/')
import torch
import numpy as np
import transforms_data as td
from PIL import Image
import glob
from time import sleep
from torch import cuda
import acquire_ear_dataset as a
import os
import shutil
from DLBio.pytorch_helpers import get_device

# Pin imports
from gpiozero import LED
import RPi.GPIO as GPIO
from Adafruit_CharLCD import Adafruit_CharLCD




CATEGORIES = ["mila_wol", "falco_len", "jesse_kru", "konrad_von", "nils_loo", "johannes_boe", "johannes_wie", "sarah_feh", "janna_qua", "tim_moe"]
CATEGORIES.sort()
AUTHORIZED = ["falco_len","konrad_von"]
RESIZE_Y = 150
RESIZE_X = 100
DATA_TEST_FOLDER = "../auth_dataset/unknown-auth/*png"
DEVICE = get_device()

model = torch.load('./class_sample/model.pt', DEVICE)

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
    access = False
    for a in authentification_dict:
        if a in AUTHORIZED and summ_pred[0][CATEGORIES.index(a)]>= NUMBER_AUTHORIZED:
            
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
            
            print("Access granted! Welcome "  + a + "!")
            led_green.on()
            sleep(10)
            led_green.off()
            access = True
            break

    if not access :
        # LCD output
        lcd.clear()
        lcd.message('Access denied -\nNo entry.')

        print("Access Denied")
        led_red.on()
        sleep(10)
        led_red.off()

finally:
    # clear outputs
    lcd.clear()
#     GPIO.cleanup()

    # %%
    shutil.rmtree('../auth_dataset/unknown-auth')
