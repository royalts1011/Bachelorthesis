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
from pickle import UnpicklingError
import torchvision

class TestMyModel:

    DEVICE = get_device()
    PATH_TO_MODELS = './class_sample/'

    def __init__(self, data_loader_test):
        self.dl_test = data_loader_test
        self.CATEGORIES = data_loader_test.dataset.classes
        # CATEGORIES.sort()

    # method for displaying files with index
    def print_list(self, list_):
        fmt = '{:<8}{:<20}'
        print(fmt.format('Index', 'Name'))
        for i, name in enumerate(list_):
            print(fmt.format(i, name))


    def load_model(self):
        models = os.listdir(self.PATH_TO_MODELS)
        self.print_list(models)
        # Handle the user's input for the model to load
        while True:
            idx = input('Choose the model by index: ')
            try:
                idx = int(idx)
                assert idx < len(models) and idx >= 0
                break
            except (ValueError, AssertionError):
                print('The input was a string or not in the index range.')
        
        # Handle the model loading
        try:
            self.model = torch.load(os.path.join(self.PATH_TO_MODELS, models[idx]), self.DEVICE)
        except (UnboundLocalError, UnpicklingError, IndexError):
            print('The chosen file ', models[idx], ' might be corrupted or not a model at all.')
        
        print(self.model.eval())


    def begin_testing(self):
        assert hasattr(self, 'model'), 'Model was not loaded!'
        data_iter = iter(self.dl_test)
        images, labels = data_iter.next()
        # outputs = self.model(images)

        all_classes = []
        for i in images:
            i = torch.unsqueeze(i, 0)
            with torch.no_grad():
                pred = self.model(i)
                pred = torch.softmax(pred, 1)
                pred = pred.cpu().numpy()

            classes = np.argmax(pred, classes)
            all_classes.append(classes[0])
            pred = np.append(pred, classes)
            pred = np.append(pred, self.CATEGORIES[classes[0]])	
            print(pred, "\n")
        
        print(all_classes)


    def start_testing(self):
        assert hasattr(self, 'model'), 'Model was not loaded!'
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.dl_test:
                images, labels = data
                # for i in images:
                    # i = i.permute(3, 0, 1, 2)
                    # if cuda.is_available():
                    #     i = i.type('torch.cuda.FloatTensor')
                    # else:
                    #     i = i.type('torch.FloatTensor')
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

