import torch.nn as nn
import cv2
from torchvision import transforms
import torch
import torchvision
import numpy as np
import ds_ear
import glob
from PIL import Image
from matplotlib import image
import sys
sys.path.append('../..')
from DLBio import pt_training
from torchvision.models.mobilenet import mobilenet_v2
from DLBio.pytorch_helpers import get_device, get_num_params
from DLBio.helpers import check_mkdir
from DLBio.pt_train_printer import Printer
import json
import matplotlib.pyplot as plt
from os.path import join


CATEGORIES = ["Konrad", "Falco"]
RESIZE_Y = 150
RESIZE_X = 100
DATA_TEST_FOLDER = "../test/*png"


def get_data(folder):
    img_array = []
    img_array_resized = []
    files = glob.glob (folder)
    for idx, f in zip(range(len(files)),files):
        image = cv2.imread(f)
        img_array.append (image)
        img_array_resized.append(cv2.resize(img_array[idx],(RESIZE_Y,RESIZE_X)))
    return np.asarray(img_array_resized)


model = torch.load('./class_sample/model.pt')

data = get_data(DATA_TEST_FOLDER)
data_tensor = torch.from_numpy(data)
data_tensor = data_tensor.permute(0, 3, 1, 2)
data_tensor = data_tensor.type('torch.cuda.FloatTensor')



NUMBER_AUTHORIZED = int(.7*len(data_tensor))

with torch.no_grad():
	# pred = model(image_transformed)
	pred = model(data_tensor)
	pred = torch.softmax(pred, 1)
	pred = pred.cpu().numpy()

classes_ = np.argmax(pred, 1)
print(pred)
print(classes_)
counts = np.bincount(classes_)
if np.max(counts) > NUMBER_AUTHORIZED:
	print("Welcome to your save room " + CATEGORIES[np.argmax(counts)] + "!")
else: 
	print("Authentification Failed! You got no acces rights!")