import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import numpy as np
import os


def get_origin_data():
    X = []
    Y = []
    path = 'dataset/DataGood/origin'
    for dir_path, dir_names, file_names in os.walk(path):
        for file_name in file_names:
            # Using os.path.join to correctly handle file paths
            file_path = os.path.join(dir_path, file_name)
            array = np.array(Image.open(file_path), dtype=np.uint8)
            X.append(array)
            label_y = int(file_name.split('.')[-2])
            Y.append(label_y)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx])
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def inception_residual_network(pretrained=True, num_classes=24):
    model = models.inception_v3(pretrained=pretrained, aux_logits=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)

        print(f'Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model


def run_training():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = inception_residual_network().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load data and create dataloaders
    X, Y = get_origin_data()  # Make sure to implement or modify this function based on the dataset
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(X, Y):
        train_dataset = CustomDataset(X[train_index], Y[train_index], transform=get_transforms())
        test_dataset = CustomDataset(X[test_index], Y[test_index], transform=get_transforms())

        train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=40, shuffle=False)

        dataloaders = {'train': train_loader, 'val': test_loader}

        trained_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)
        # Save or evaluate the model as needed


if __name__ == '__main__':
    run_training()
