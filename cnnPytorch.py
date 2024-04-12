import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# Load the pre-trained ResNet-50 model
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace the top layer for fine-tuning
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 50),
    nn.Softmax(dim=1)
)

# Define transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder('Dataset/DataGood/ChromoClassified/', transform=train_transform)
valid_dataset = ImageFolder('Dataset/DataGood/ChromoClassified/', transform=valid_transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.1)


# Training loop
def train_model(model, criterion, optimizer, num_epochs=10):
    model.to(device)
    for epoch in tqdm(range(num_epochs), desc='Epoch Progress', unit='epoch'):
        model.train()  # Set model to training mode
        train_loss = 0
        train_progress = tqdm(train_loader, desc='Training Batch', leave=False)
        for inputs, labels in train_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()  # Set model to evaluate mode
        valid_loss = 0
        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader, desc='Validation Batch', leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(valid_loader)
        print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}')


# Specify the device
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Start the training process
train_model(model, criterion, optimizer)
