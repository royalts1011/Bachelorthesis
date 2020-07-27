import torchvision
import transforms_data as td
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from siamese_network_dataset import SiameseNetworkDataset


def get_dataloader(data_path, indices, transformation, batch_size=32, num_workers=0, should_invert = False):

    # indices limit the range that images are randomly picked from
    siam_dset = get_siam_dataset(data_path, indices, transformation, should_invert)

    # load the data into batches
    data_loader = DataLoader(
        siam_dset,
        batch_size=batch_size,
        # reshuffle after every epoch
        shuffle=False,
        num_workers=num_workers
    )
  
    return data_loader


def get_dataset(data_path, transformation):
    
    # create dataset with dict transformation
    dataset = torchvision.datasets.ImageFolder(
                    root = data_path,
                    transform = transformation
                    ) 

    print(dataset.classes)
    return dataset

def get_siam_dataset(data_path, indices, transformation, should_invert):

    # loads dataset from disk
    dataset = torchvision.datasets.ImageFolder( 
                    root = data_path
                    )

    # uses custom dataset class to create a siamese dataset
    siamese_dataset = SiameseNetworkDataset(
                        imageFolderDataset = dataset,
                        indices=indices,
                        transform=transformation,
                        should_invert=should_invert)
    
    return siamese_dataset
