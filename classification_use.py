from torchvision.models.mobilenet import mobilenet_v2
import torch.nn as nn
import cv2
from torchvision import transforms
import torch
import torchvision
import numpy as np

CATEGORIES = ["Konrad", "Falco"]
RESIZE_M = 150
RESIZE_N = 100



def get_data(file):
    img_array = cv2.imread(file, cv2.IMREAD_COLOR)
    img_array_resized = cv2.resize(img_array,(RESIZE_M,RESIZE_N))
    return img_array_resized.reshape(-1, RESIZE_M, RESIZE_N, 1)

model = torch.load('/nfshome/lentzsch/Documents/Bachelorarbeit/Bachelorthesis/class_sample/model.pt')

data = get_data('/nfshome/lentzsch/Documents/Bachelorarbeit/test/konrad001.png')
data_tensor = torch.from_numpy(data)
data_tensor = data_tensor.permute(3, 0, 1, 2)
data_tensor = data_tensor.type('torch.cuda.FloatTensor')

with torch.no_grad():
	pred = model(data_tensor)
	pred = torch.softmax(pred, 1)
	pred = pred.cpu().numpy()

classes_ = np.argmax(pred, 1)
print(classes_)
a=1