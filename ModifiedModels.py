import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from torchvision.models import Inception_V3_Weights
from tqdm import tqdm


class ModifiedInceptionV3Paper(nn.Module):
    def __init__(self, num_classes=24) -> None:
        super(ModifiedInceptionV3Paper, self).__init__()
        self.inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception.Conv2d_1a_3x3.conv = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.inception.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        self.name = "V3ModInceptionPaper"
        #additional layers
        self.dropout = nn.Dropout(0.5)
        self.global_average_pooling2d = nn.AvgPool2d()
        self.dense = nn.Linear(2048, 256)
        self.batch_norm_layer = nn.BatchNorm1d(num_features=256)
        self.output = nn.Linear(256, num_classes)

    def forward(self,x):
        x = self.inception(x)

        x = self.dropout(x)
        x = self.global_average_pooling2d(x)
        x = self.dense(x)
        x = self.batch_norm_layer(x) 
        x = self.output(x)
    

class ModifiedInceptionV3(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedInceptionV3, self).__init__()
        self.inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception.Conv2d_1a_3x3.conv = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.inception.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        self.name = "V3ModInception"

    def forward(self, x):
        return self.inception(x)
    

class ModifiedResNet(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedResNet, self).__init__()
        
        resnet50 = models.resnet50(pretrained=True)
        num_features = resnet50.fc.in_features
        resnet50.fc = nn.Linear(num_features, num_classes)
        for param in resnet50.parameters():
            param.requires_grad = False
        
        self.model = resnet50

    def forward(self, x):
        return self.model(x)

