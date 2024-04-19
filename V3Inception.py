import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import Inception_V3_Weights
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

model_name = "V3Inception.pt"
data_dir = 'dataset/DataGood/ChromoClassified'
# data_dir = 'Dataset/Data/24_chromosomes_object/preprocessed_images'  # Update with your dataset directory

num_classes = 24  # Update with the number of chromosome classes

# Load pre-trained InceptionV3 model
inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)

# Modify the last layer for your specific task TODO need to improve
num_ftrs = inception.fc.in_features
# num_classes is the number of chromosome classes
inception.fc = torch.nn.Linear(num_ftrs, num_classes)

# Define your loss function
criterion = nn.CrossEntropyLoss()

# Define your optimizer
optimizer = optim.Adam(inception.parameters(), lr=0.0001)  # looks good for all

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

inception.to(device)

inception.train()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=5, translate=(
            0.05, 0.05), scale=(0.95, 1.05), shear=5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    ], p=0.9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
])

# Load dataset and apply transformations
full_dataset = ImageFolder(root=data_dir, transform=transform)

# Train-test-validation split
train_size = int(0.7 * len(full_dataset))
validation_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - validation_size
train_dataset, validation_dataset, test_dataset = random_split(
    full_dataset, [train_size, validation_size, test_size])

# DataLoader setup
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(
    validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize lists for storing metrics
train_losses, train_accuracies = [], []
validation_losses, validation_accuracies = [], []
test_losses, test_accuracies = [], []

best_validation = 0.0
# Training loop
num_epochs = 5
for epoch in tqdm(range(num_epochs), desc='Epoch Progress'):
    inception.train()
    running_loss, correct_predictions, total_predictions = 0.0, 0, 0

    # Training phase
    for inputs, labels in tqdm(train_loader, desc='Training Batch', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, aux_outputs = inception(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        # tensors = vectors of 0 and 1 is true
        correct_predictions += (predicted == labels).sum().item()
        # for parallelization of neural network
        total_predictions += labels.size(0)
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(correct_predictions / total_predictions)

    # Validation phase
    inception.eval()
    validation_loss, correct_val, total_val = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = inception(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    validation_losses.append(validation_loss / len(validation_loader))
    validation_accuracies.append(correct_val / total_val)

    if correct_val/total_val > best_validation:
        best_validation = correct_val/total_val
        torch.save(inception.state_dict(), "models/" + model_name)
    print('Best validation accuracy: ', best_validation)


inception.load_state_dict(torch.load("models/" + model_name))
inception.eval()

# Testing phase again

test_loss, correct_test, total_test = 0.0, 0, 0
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Testing Batch'):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = inception(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_test += (predicted == labels).sum().item()
        total_test += labels.size(0)

print(correct_test / total_test)


# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss across Training, Validation, and Testing')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(validation_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy across Training, Validation, and Testing')
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('model_performance_metrics.png')