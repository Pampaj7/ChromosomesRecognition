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
import json


def chooseModel(model_type):
    if model_type == 0:
        return ModifiedInceptionV3(24)
    elif model_type == 1:
        return ModifiedInceptionV3Paper(24)
    elif model_type == 2:
        return ModifiedVGG16(24)
    elif model_type == 3:
        return ModifiedResNet50(24)
    else:
        raise ValueError("Unknown model_type")


# took input 224x224
def processData(batch_size, modelname):
    # create the data loader to use during train
    if modelname == "V3ModInceptionPaper" or modelname == "V3ModInception":
        pad = (299 - 224) // 2
        res = 299  # needed cause the approximation of the inception model
    else:
        pad = 0
        res = 224
    transform = transforms.Compose([
        transforms.Pad((pad, pad), fill=0, padding_mode='constant'),
        transforms.Resize(res),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        ], p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
    ])

    full_dataset = ImageFolder('dataset/DataGood/ChromoClassified', transform=transform)

    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train(model, train_loader, val_loader, test_loader, lr, epochs, opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = opt(model.parameters(), lr=lr, weight_decay=0.5)  # weight decay is a regularization term
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # multiplied by 0.1 each 30 epochs

    early_stop_counter = 0
    patience = 5  # epochs to wait before stopping

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
            if model.name == "VGG16" or model.name == "ResNet50":  # name defined in the ModifiedModels.py
                outputs = model(images)  # if vgg16 and resnet need 1 parameter
            else:
                outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

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
            early_stop_counter = 0  # reset the counter
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%')

    metrics = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'validation_losses': validation_losses,
        'validation_accuracies': validation_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    }
    with open(f'models/{model.name}_metrics.json', 'w') as f:
        json.dump(metrics, f)

    return train_losses, train_accuracies, validation_losses, validation_accuracies, test_losses, test_accuracies


def plot(model, train_losses, train_accuracies, validation_losses, validation_accuracies, test_losses, test_accuracies):
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
    plt.savefig('models/' + model.name + ".png")
    plt.show()


def model_summary(model):
    print("Model Summary:")
    total_params = 0
    for name, layer in model.named_modules():
        # Skip the top-level module (the whole model)
        if isinstance(layer, nn.Module) and not isinstance(layer, nn.Sequential) and layer != model:
            params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            print(
                f"{name.ljust(20)} | {str(layer).ljust(30)} | Params: {params} | Trainable: {any(p.requires_grad for p in layer.parameters())}")
            total_params += params
    print(f"Total trainable parameters: {total_params}")


def load_all_metrics(model_name):
    with open(f'models/{model_name}_metrics.json', 'r') as f:
        metrics = json.load(f)
    return metrics


def final_plot():
    # Names of the models you have saved metrics for
    model_names = ["V3ModInception",
                   # "V3ModInceptionPaper",
                   "VGG16",
                   "ResNet50"]
    metrics_labels = ['train_losses',
                      'train_accuracies',
                      'validation_losses',
                      'validation_accuracies',
                      'test_losses',
                      'test_accuracies']

    # Load metrics for each model and prepare the plots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2x3 grid for six metrics
    fig.suptitle('Comparison of Training Metrics Across Models')

    for i, metric_label in enumerate(metrics_labels):
        ax = axs[i // 3, i % 3]  # Determine the subplot position
        for name in model_names:
            metrics = load_all_metrics(name)
            ax.plot(metrics[metric_label], label=name)
            ax.set_title(metric_label.replace('_', ' ').capitalize())
            ax.set_xlabel('Epochs')
            ax.set_ylabel(metric_label.split('_')[1].capitalize())
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def pipeline(model_type):
    model = chooseModel(model_type)
    # model_summary(model)
    train_loader, val_loader, test_loader = processData(batch_size=16, modelname=model.name)
    train_losses, train_accuracies, validation_losses, validation_accuracies, test_losses, test_accuracies = train(
        model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, lr=0.0001, epochs=30)

    plot(model, train_losses, train_accuracies, validation_losses, validation_accuracies, test_losses, test_accuracies)


def grid_search(model_type, lr_options, optimizers):
    results = {}
    for lr in lr_options:
        for optimizer in optimizers:
            print(f"Training model with lr={lr} and optimizer={optimizer}")
            model = chooseModel(model_type)
            train_loader, val_loader, test_loader = processData(batch_size=16, modelname=model.name)
            train_losses, train_accuracies, validation_losses, validation_accuracies, test_losses, test_accuracies = train(
                model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, lr=lr,
                epochs=5, opt=optimizer)

            plot(model, train_losses, train_accuracies, validation_losses, validation_accuracies, test_losses,
                 test_accuracies)

            # Store or print results
            key = f"LR: {lr}, Optimizer: {optimizer.__name__}"
            results[key] = {
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'validation_losses': validation_losses,
                'validation_accuracies': validation_accuracies
            }

    return results


lr_options = [0.0001, 0.001, 0.01]
optimizers = [optim.Adam, optim.SGD, optim.RMSprop]
"""
pipeline(0)
# pipeline(1) # TODO need to fix the model
pipeline(2)
pipeline(3)
"""
grid_search(0, lr_options, optimizers)

final_plot()
