import torchvision
import transforms_data as td
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset

DATA_FOLDER = '../AMI'
RESIZE_Y = 150
RESIZE_X = 100

def get_dataloader(indeces=None, batch_size=32, num_workers=0, is_train = True, data_path=DATA_FOLDER):

    dataset = get_dataset(data_path, is_train)

    if indeces is None:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=SubsetRandomSampler(indeces)
        )
    return data_loader


def get_dataset(data_path=DATA_FOLDER, is_train=True):

    transform_dict = {
        'train': td.transforms_train(
                    (RESIZE_Y, RESIZE_X),
                    [0.49139968, 0.48215841, 0.44653091],
                    # alternate mean: [0.485, 0.456, 0.406],
                    [0.24703223, 0.24348513, 0.26158784]
                    # alternate std: [0.229, 0.224, 0.225]
                    ),
        'valid_and_test': td.transforms_valid_and_test(
                    (RESIZE_Y, RESIZE_X),
                    [0.49139968, 0.48215841, 0.44653091],
                    # alternate mean: [0.485, 0.456, 0.406],
                    [0.24703223, 0.24348513, 0.26158784]
                    # alternate std: [0.229, 0.224, 0.225]
                    )
        }
    
    dataset = torchvision.datasets.ImageFolder(
                    root = data_path,
                    transform=transform_dict['train' if is_train else 'valid_and_test']
                    ) 

    return dataset