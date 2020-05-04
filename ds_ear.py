import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset


DATA_FOLDER = './ear-dataset-py'

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
    dataset = 


    return dataset