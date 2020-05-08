import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset

DATA_FOLDER = '../AMI'
RESIZE_M = 150
RESIZE_N = 100

def get_dataloader(indeces=None, batch_size=32, num_workers=0,data_path=DATA_FOLDER):

    dataset = get_dataset(data_path)
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


def get_dataset(data_path=DATA_FOLDER):
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([
            transforms.Resize((RESIZE_M, RESIZE_N)),
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.RandomPerspective(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.49139968, 0.48215841, 0.44653091],
                [0.24703223, 0.24348513, 0.26158784]
            )
        ])
    )
    print(dataset.classes)
    return dataset