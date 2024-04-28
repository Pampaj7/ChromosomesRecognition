import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import Inception_V3_Weights


class ModifiedInceptionV3Paper(nn.Module):
    def __init__(self, num_classes=24):
        super(ModifiedInceptionV3Paper, self).__init__()
        self.inception = models.inception_v3(pretrained=True)
        # Modify layers as needed
        self.inception.Conv2d_1a_3x3.conv = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.inception.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        # additional layers
        self.dropout = nn.Dropout(0.5)
        self.global_average_pooling2d = nn.AdaptiveAvgPool1d((1))
        self.dense = nn.Linear(2048, num_classes)  # Adjusted input size to match global average pooling
        self.batch_norm_layer = nn.BatchNorm1d(num_features=256)
        self.output = nn.Linear(256, num_classes)
        self.name = "V3ModInceptionPaper"


    def forward(self, x):
        x, aux = self.inception(x)
        x = self.dropout(x)
        x = self.global_average_pooling2d(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dense(x)
        x = self.batch_norm_layer(x)
        x = self.output(x)
        return x



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


class ModifiedVGG16(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedVGG16, self).__init__()
        # Load the pretrained VGG16 model
        original_model = models.vgg16(pretrained=True)
        self.features = original_model.features
        self.name = 'VGG16'

        # Define the classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((1, 1)),  # This is equivalent to Keras's GlobalAveragePooling2D
            nn.Flatten(),  # Flatten the output of the pooling layer
            nn.Linear(512, 256),  # Dense layer
            nn.ReLU(True),
            nn.BatchNorm1d(256),  # Batch normalization
            nn.Dropout(0.5),  # Dropout
            nn.Linear(256, num_classes),  # Output layer, replace num_classes with the actual number of classes
        )

    def forward(self, x):
        x = self.features(x)  # Pass input through feature extractor
        x = self.classifier(x)
        return x


class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedResNet50, self).__init__()
        original_model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(original_model.children())[:-1])  # Remove last layer
        self.classifier = nn.Linear(2048, num_classes)
        self.name = 'ResNet50'

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x
    
class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedResNet18, self).__init__()
        original_model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(original_model.children())[:-1])  # Remove last layer
        self.classifier = nn.Linear(512, num_classes)  # Adjust for ResNet-18
        self.name = 'ResNet18'

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x
