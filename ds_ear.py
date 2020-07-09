import torchvision
import transforms_data as td
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset

# dictionary to access different transformation methods
transform_dict = {
    'train': td.transforms_train( td.get_resize(small=False) ),
    'train_gray': td.transforms_train_grayscale( td.get_resize(small=False) ),
    'valid_and_test': td.transforms_valid_and_test( td.get_resize(small=False) ),
    'size_only' : None
}

def get_dataloader(data_path, indices=None, batch_size=32, num_workers=0, transform_mode='train'):

    dataset = get_dataset(data_path, transform_mode)

    if indices is None:
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
            sampler=SubsetRandomSampler(indices)
        )
    return data_loader


def get_dataset(data_path, transform_mode):
    
    dataset = torchvision.datasets.ImageFolder(
                    root = data_path,
                    transform=transform_dict[transform_mode]
                    ) 

    return dataset