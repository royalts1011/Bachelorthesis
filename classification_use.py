# %%
import torch
import numpy as np
import transforms_data as td
from PIL import Image
import glob
from torch import cuda
import acquire_ear_dataset as a
import os
import shutil



CATEGORIES = ["falco_lem", "jesse_kru", "konrad_von", "nils_loo", "johannes_boe", "johannes_wie", "sarah_feh"]
CATEGORIES.sort()
AUTHORIZED = ["Falco","Konrad"]
RESIZE_Y = 150
RESIZE_X = 100
DATA_TEST_FOLDER = "../auth_dataset/unknown-auth/*png"

model = torch.load('./class_sample/model.pt')


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
# Bilder aufnehmen
a.capture_ear_images(amount_pic=10, pic_per_stage=10, is_authentification=True)


# %%
all_classes = []
summ_pred = np.empty(1)
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


# %%
# Hier besser die Warscheinlichkeit über alle Bilder ermitteln und darüber prüfen.
# Beispiel: Bei 5 Bilder muss die aufsummierte Wahrscheinlichkeit für eine Person >4 sein!!

NUMBER_AUTHORIZED = int(.7*len(image_array))
authentification_dict = {CATEGORIES[i]:all_classes.count(i) for i in all_classes}
print(authentification_dict) 

for a in authentification_dict:
    if a in AUTHORIZED and summ_pred[0][CATEGORIES.index(a)]>= NUMBER_AUTHORIZED:
        print("Access granted! Welcome "  + a + "!")
        break
    else:
        print("Access denied")
        break


# %%
shutil.rmtree('../auth_dataset/unknown-auth')

