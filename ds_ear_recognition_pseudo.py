from torch.utils.data import DataLoader, Dataset
import torch
from torchvision
import torchvision.transforms
from os.path import splitext, join
import re

IMAGE_NAME_RGX = r’(.*)_\d{3}'

LABEL_INDECES = {‚falco‘:1, ‚konrad‘:2} # other -> 0

def get_dataloader(batch_size, num_workers, **kwargs):
    dataset = EarRecognition(**kwargs)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, …)

class EarRecognition(Dataset):
    def __init__(self, path=PATH_TO_IMAGES, **kwargs):
        self.images, self.labels = self._load_images(path)
        self.to_tens = torchvision.transforms.ToTensor()
    
    def __getitem__(self, index):
        image ,label = self.images[index], self.labels[index]

        image = self.to_tens(image)
        label = torch.tensor(label).long()
        return image, label

    def __len__(self):
        return len(self.images)

    def _load_images(self, path):
        self.images = []
        self.labels = []
        for root, _, files_ in os.walk(path):
        files_ = [x for x in files_ if splitext(x)[-1] == „.png“]
        if not files_:
        continue

        for file in files_:
        self.images.append(cv2.imread(join(root, file)))
        # derive label from filename
        match = re.match(IMAGE_NAME_RGX, file)
        assert bool(match), f’file {file} does not conform to name convention’
        name = match.group(1)

        label = LABEL_INDECES.get(name, 0) # zero if no fit
        self.labels.append(label)
        # make sure the found images make sense
        assert self.images, f’no images found in {path}'
        assert len(set(self.labels)) > 1, ‚dataset has less than two classes‘