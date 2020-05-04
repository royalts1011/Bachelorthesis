import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset


DATA_FOLDER = 'dataset'
RESIZE_M = 150
RESIZE_N = 100

def get_dataloader(
    is_train=True, indeces=None, batch_size=32, num_workers=0,
        data_path=DATA_FOLDER):
    # indeces, if you only want to train a subset
    dataset = get_dataset(is_train, data_path)

    if indeces is None:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            sampler=SubsetRandomSampler(indeces)
        )
    return data_loader


def get_dataset(is_train=True, data_path=DATA_FOLDER):
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
    print("helloow")
    print(dataset.classes)

    return dataset