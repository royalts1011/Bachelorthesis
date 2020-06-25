import torchvision
import transforms_data as td
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from siamese_network_dataset import SiameseNetworkDataset


# dictionary to access different transformation methods
transform_dict = {
    'train': td.transforms_train( td.get_resize(small=False) ),
    'valid_and_test': td.transforms_valid_and_test( td.get_resize(small=False) ),
    'siamese' : td.transforms_siamese( td.get_resize(small=True) ),
    'size_only' : None
}


def get_siam_dataloader(data_path, batch_size=32, num_workers=0, transform_mode = 'train', should_invert = False):
    # get the specific siamese dataset
    siam_dset = get_siamese_dataset(data_path, transform_mode, should_invert)

    data_loader = DataLoader(
        siam_dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return data_loader



def get_siamese_dataset(data_path, transform_mode = 'siamese', should_invert = False ):
    # loads dataset from disk
    dataset = torchvision.datasets.ImageFolder( root = data_path )
    # uses custom dataset class to create a siamese dataset
    siamese_dataset = SiameseNetworkDataset(
                        imageFolderDataset = dataset,
                        transform = transform_dict[transform_mode], # applies transformation on images
                        should_invert=False)
    
    return siamese_dataset
