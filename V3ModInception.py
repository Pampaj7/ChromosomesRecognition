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
from ModifiedModels import *


def chooseModel(model):
    #
    # 0 = ModifiedInceptionV3
    # 1 = ModifiedInceptionV3Paper
    if( model == 0):
        return ModifiedInceptionV3(24)
    elif( model == 1):
        return ModifiedInceptionV3Paper(24)
    

def processData(batch_size):
    #create the data loader to use during train
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        ], p=0.9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
    ])

    full_dataset = ImageFolder('dataset/DataGood/ChromoClassified', transform=transform)
    full_dataset.class_to_idx
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train(model, train_loader, val_loader, test_loader, lr, epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize lists for storing metrics
    train_losses, train_accuracies = [], []
    validation_losses, validation_accuracies = [], []
    test_losses, test_accuracies = [], []

    best_validation = 0
    num_epochs = epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate training metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # Calculate validation metrics
        validation_losses.append(val_loss / len(val_loader))
        validation_accuracies.append(correct_val / total_val)

        # Test set evaluation
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_losses.append(test_loss / len(test_loader))
        test_accuracies.append(correct_test / total_test)

        # Update best validation
        if correct_val / total_val > best_validation:
            best_validation = correct_val / total_val
            torch.save(model.state_dict(), "models/" + model.name + ".pt")

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%')

    return  train_losses, train_accuracies, validation_losses, validation_accuracies, test_losses, test_accuracies    


def plot(model,train_losses, train_accuracies, validation_losses, validation_accuracies, test_losses, test_accuracies):
# Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss across Training, Validation, and Testing')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(validation_accuracies, label='Validation Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy across Training, Validation, and Testing')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('models/' + model.name +".png")
    plt.show()

def pipeline(model_type):
    model = chooseModel(0)
    train_loader, val_loader, test_loader = processData(batch_size = 64)
    train_losses, train_accuracies, validation_losses, validation_accuracies, test_losses, test_accuracies = train(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, lr=0.0001, epochs=150)
    plot(model,train_losses, train_accuracies, validation_losses, validation_accuracies, test_losses, test_accuracies)


pipeline(0)


