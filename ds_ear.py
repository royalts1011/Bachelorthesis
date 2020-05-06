import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
import numpy as np
from os.path import join, dirname
import os


DATA_FOLDER = join(dirname(os.getcwd()),'AMI')
RESIZE_M = 150
RESIZE_N = 100

def get_dataloader(
    is_train=True, indeces=None, batch_size=32, num_workers=0,
        data_path=DATA_FOLDER):
    # indeces, if you only want to train a subset
    dataset = get_dataset(data_path)

    # Split dataset into fix train and test
    train_size = 0.6
    dataset_size = len(dataset)
    # split all indices at train percentage
    indices = list(range(dataset_size))
    split = int(np.floor(train_size * dataset_size))
    np.random.seed(0)
    np.random.shuffle(indices)

    train_indices, test_indices = indices[:split], indices[split:]
    # create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    if is_train:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,#args.batch_size,
            num_workers=num_workers,#args.loader_num_workers,
            sampler=train_sampler)
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,#args.batch_size,
            num_workers=num_workers,#args.loader_num_workers,
            sampler=test_sampler)

    # if indeces is None:
    #     data_loader = DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         shuffle=is_train,
    #         num_workers=num_workers
    #     )
    # else:
    #     data_loader = DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         shuffle=is_train,
    #         num_workers=num_workers,
    #         sampler=SubsetRandomSampler(indeces)
    #     )
    return data_loader

def get_dataset(data_path=DATA_FOLDER):
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((RESIZE_M, RESIZE_N)),
            torchvision.transforms.Lambda(lambda x: x.convert('RGB')),
            torchvision.transforms.ToTensor(),

            torchvision.transforms.Normalize(
                [0.49139968, 0.48215841, 0.44653091],
                [0.24703223, 0.24348513, 0.26158784]
            )   
        ])
        
    )
    print(dataset.classes)
    return dataset