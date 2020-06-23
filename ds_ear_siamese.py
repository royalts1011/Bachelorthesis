import torchvision
import transforms_data as td
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from siamese_network_dataset import SiameseNetworkDataset
from siamese_verification_dataset import SiameseVerificationDataset


DATA_FOLDER = '../AMIC'
RESIZE_BIG = (224, 224)
RESIZE_SMALL = (100, 100)

# dictionary to access different transformation methods
transform_dict = {
    'train': td.transforms_train( RESIZE_BIG ),
    'valid_and_test': td.transforms_valid_and_test( RESIZE_BIG ),
    'siamese' : td.transforms_siamese( RESIZE_SMALL ),
    'size_only' : None
}


def get_dataloader(indices=None, batch_size=32, num_workers=0, transform_mode = 'train', data_path=DATA_FOLDER, should_invert = False):

    siam_dset = get_siamese_dataset(data_path, transform_mode, should_invert)

    if indices is None:
        data_loader = DataLoader(
            siam_dset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
    else:
        data_loader = DataLoader(
            siam_dset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=SubsetRandomSampler(indices)
        )
    return data_loader


def get_dataset(data_path=DATA_FOLDER, transform_mode = 'size_only'):
    
    # create dataset with dict transformation
    dataset = torchvision.datasets.ImageFolder(
                    root = data_path,
                    transform=transform_dict[transform_mode]
                    ) 

    print(dataset.classes)
    return dataset

def get_siamese_dataset(data_path=DATA_FOLDER, transform_mode = 'siamese', should_invert = False ):

    # loads dataset from disk
    dataset = torchvision.datasets.ImageFolder( root = data_path )
    # uses custom dataset class to create a siamese dataset
    siamese_dataset = SiameseNetworkDataset(
                        imageFolderDataset = dataset,
                        transform = transform_dict[transform_mode], # applies transformation on images
                        should_invert=False)
    
    return siamese_dataset

def get_verification_dataset(new_images_path, dataset_path, folder_comparison, transform_mode = 'siamese', should_invert = False ):
    # the newly taken images of the person
    new_img_dset = torchvision.datasets.ImageFolder( root = new_images_path )
    general_dataset = torchvision.datasets.ImageFolder( root = dataset_path )

    # uses custom dataset class to create a siamese dataset
    verification_dataset = SiameseNetworkDataset(
                        newimageDataset = new_img_dset,
                        generalDataset = general_dataset,
                        folder_comparison = folder_comparison,
                        transform = transform_dict[transform_mode], # applies transformation on images
                        should_invert=False)
    
    return verification_dataset
