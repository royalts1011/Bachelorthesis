import torch.nn as nn
from torchvision.models.mobilenet import MobileNetV2

class MyMobileNetV2(MobileNetV2):

    def __init__(self):
        super(MyMobileNetV2, self).__init__()
        self.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.classifier[1] = nn.Linear(in_features=self.classifier[1].in_features, out_features=5)

