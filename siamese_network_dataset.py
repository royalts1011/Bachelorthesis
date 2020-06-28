from torch.utils.data import DataLoader,Dataset
import random
import numpy as np
from PIL import Image
import torch
import PIL.ImageOps  
class SiameseNetworkDataset(Dataset):
    """
    Class for constructing the Siamese dataset.
    The class constructs bundles/tupples of two images and a binary label. If the images of one tuple are the same, the label is set to zero
    if they are different the label is set to 1.
 
    Initialize with the path to the dataset and indices if the dataset should be limited to a certain range of images.

    """
    def __init__(self, imageFolderDataset, indices, transform, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        # if no indices are given return all indices for later random choice
        if indices is None: self.indices = list(range(len(imageFolderDataset)))
        else: self.indices = indices

        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        # imageFolderDataset.imgs contains all image paths of the complete dataset
        # random.choice() leads to a random pick of an image path from the indices
        img0_tuple = self.imageFolderDataset.imgs[random.choice(self.indices)]

        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = self.imageFolderDataset.imgs[random.choice(self.indices)]
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found
                img1_tuple = self.imageFolderDataset.imgs[random.choice(self.indices)] 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        # Grey conversion
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)
        
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        # The indices are either a preset range or a list of all image index entries
        return len(self.indices)
