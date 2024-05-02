from matplotlib import pyplot as plt
from torch.utils.data import random_split
import ModifiedModels as mm
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile
from tqdm import tqdm
from torch.utils.data import DataLoader  # Add this import

ImageFile.LOAD_TRUNCATED_IMAGES = True

classes = 24


def processData(batch_size, modelname):
    # create the data loader to use during train
    if modelname == "V3ModInceptionPaper" or modelname == "V3ModInception":
        pad = (299 - 224) // 2
        res = 299  # needed cause the approximation of the inception model
    else:
        pad = 0
        res = 224
    transform = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        ], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
    ])

    full_dataset = datasets.ImageFolder(root='DataNew/', transform=transform)

    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train(model, train_loader, val_loader, test_loader, lr, epochs, opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = opt(model.parameters(), lr=lr, weight_decay=1e-5)  # weight decay is a regularization term
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)  # multiplied by 0.1 each 30 epochs

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
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if model.name in ["VGG16", "ResNet50"]:
                outputs = model(images)
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
        valLoss = val_loss / len(val_loader)
        valAcc = correct_val / total_val

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
            torch.save(model.state_dict(), "models/Test" + model.name + ".pt")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {valLoss:.4f}, Accuracy: {valAcc * 100:.2f}%')

    return train_losses, train_accuracies, validation_losses, validation_accuracies, test_losses, test_accuracies


def plot(model, train_losses, train_accuracies, validation_losses, validation_accuracies, test_losses, test_accuracies,
         lr, optimizer_name, model_type):
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title(f'Loss | {model_type} | LR: {lr} | Optimizer: {optimizer_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(validation_accuracies, label='Validation Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title(f'Accuracy | {model_type} | LR: {lr} | Optimizer: {optimizer_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    filename = f'plots/{model.name}Test_LR{lr}_OPT{optimizer_name}.png'
    plt.savefig(filename)
    plt.show()


model = mm.ModifiedResNet50(24)
# model_summary(model)
train_loader, val_loader, test_loader = processData(batch_size=16, modelname=model.name)
train_losses, train_accuracies, validation_losses, validation_accuracies, test_losses, test_accuracies = train(
    model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, lr=0.0001, epochs=30,
    opt=optim.Adam)

plot(model, train_losses, train_accuracies, validation_losses, validation_accuracies, test_losses, test_accuracies, 0.0001, "Adam", "InceptionV3")
