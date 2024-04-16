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

batch_size = 16
data_dir = 'dataset/DataGood/ChromoClassified'  # Update with your dataset directory

# Load pre-trained InceptionV3 model
inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)

# Modify the last layer for your specific task TODO
num_ftrs = inception.fc.in_features
inception.fc = torch.nn.Linear(num_ftrs, 24)  # num_classes is the number of chromosome classes

# Define your loss function
criterion = nn.CrossEntropyLoss()

# Define your optimizer
optimizer = optim.Adam(inception.parameters(), lr=0.0001)  # looks good for all

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

inception.to(device)

inception.train()

num_epochs = 10

# Define transformations
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# Load dataset and apply transformations
full_dataset = ImageFolder(root=data_dir, transform=transform)

# Train-test-validation split
train_size = int(0.7 * len(full_dataset))
validation_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - validation_size
train_dataset, validation_dataset, test_dataset = random_split(full_dataset, [train_size, validation_size, test_size])

# DataLoader setup
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training loop
num_epochs = 10
for epoch in tqdm(range(num_epochs), desc='Epoch Progress'):
    inception.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # Training phase
    for inputs, labels in tqdm(train_loader, desc='Training Batch'):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, aux_outputs = inception(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
        running_loss += loss.item()

    # Calculate and display metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_predictions
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    # Validation phase
    inception.eval()
    validation_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = inception(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    validation_accuracy = correct_val / total_val
    print(f"Validation Loss: {validation_loss / len(validation_loader):.4f}, Accuracy: {validation_accuracy:.4f}")

# Testing phase
inception.eval()
test_loss = 0.0
correct_test = 0
total_test = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = inception(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_test += (predicted == labels).sum().item()
        total_test += labels.size(0)

test_accuracy = correct_test / total_test
print(f"Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {test_accuracy:.4f}")
