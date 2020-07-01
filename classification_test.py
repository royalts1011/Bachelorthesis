import sys
sys.path.append('../..')
import torch
import numpy as np
import transforms_data as td
from PIL import Image
import glob
import os
import shutil
from pickle import UnpicklingError

# PyTorch
from torch import cuda
import torchvision

# DLBio and own scripts
import helpers
import acquire_ear_dataset as a
from DLBio.pytorch_helpers import get_device

class TestMyModel:

    DEVICE = get_device()

    def __init__(self, data_loader_test, path_to_models):
        self.dl_test = data_loader_test
        self.CATEGORIES = data_loader_test.dataset.classes
        self.PATH_TO_MODELS = path_to_models
        # CATEGORIES.sort()

    def load_model(self):
        models = helpers.rm_DSStore( os.listdir(self.PATH_TO_MODELS) )
        helpers.print_list(models)
        # Handle the user's input for the model to load
        while True:
            idx = input('Choose the model by index: ')
            try:
                idx = int(idx)
                assert idx < len(models) and idx >= 0
                if os.path.splitext(models[idx])[1] != '.pt': raise TypeError
                break
            except (ValueError, AssertionError):
                print('The input was a string or not in the index range.')
            except TypeError:
                print('The input is not of ending ".pt"')
        
        # Load model
        print('Loading ', models[idx], ' ...')
        self.model = torch.load(os.path.join(self.PATH_TO_MODELS, models[idx]), self.DEVICE)
        
        # print(self.model.eval())


    # Compute the accuracy for the test set
    def start_testing(self):
        assert hasattr(self, 'model'), 'Model was not loaded!'
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.dl_test:
                images, labels = data
                # convert to correct type
                images = helpers.type_conversion(images)
                labels = helpers.type_conversion(labels)
                # continue prediction
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('\n Accuracy of the network : {:.0%} \n'.format(correct / total) )


    def class_acc(self):
        assert hasattr(self, 'model'), 'Model was not loaded!'
        classes = self.dl_test.dataset.classes
        class_correct = list(0. for i in range( len(classes) ))
        class_total = list(0. for i in range( len(classes) ))
        with torch.no_grad():
            for data in self.dl_test:
                images, labels = data
                # convert to correct type
                images = helpers.type_conversion(images)
                labels = helpers.type_conversion(labels)
                # continue prediction
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                labels = labels.to(dtype=torch.long, device=labels.device)
                # go through batch "self.dl_test.batch_size" (last batch might be smaller, thus current labels length)
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        # print accuracies of the classes
        for i in range( len(classes) ):
            try:
                total = 'Number of images: ' + str(int(class_total[i]))
                print('Accuracy of {:>15} : {:>6.2f} % \t{:>22}'.format(
                    classes[i], 100 * class_correct[i] / class_total[i], total))
            except ZeroDivisionError:
                print('Accuracy of {:>15} : {:<3}'.format(classes[i], "None") )