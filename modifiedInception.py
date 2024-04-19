import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models import Inception_V3_Weights
from tqdm import tqdm


# Step 1: Load pre-trained model and modify input and output layers
class ModifiedInceptionV3(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedInceptionV3, self).__init__()
        self.inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception.Conv2d_1a_3x3.conv = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.inception.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.inception(x)


# Step 2: Define data transformations THE MODEL WORKS WITH 3 CHANNELS
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),  # Convert to grayscale and expand to three channels
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    ], p=0.9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])  # Apply normalization for three channels
])

# Step 3: Load dataset
train_dataset = ImageFolder('dataset/DataGood/ChromoClassified', transform=transform)
val_dataset = ImageFolder('dataset/DataGood/ChromoClassified', transform=transform)

# Step 4: Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Step 5: Define loss function and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model = ModifiedInceptionV3(num_classes=24).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Step 6: Fine-tuning loop with tqdm
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(images)  # double output
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': running_loss / (total / train_loader.batch_size),
                              'accuracy': (correct / total) * 100})
            pbar.update(1)

    # Step 7: Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss += criterion(outputs, labels).item()

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {(100 * correct / total):.2f}%')

# Save the model's state_dict
torch.save(model.state_dict(), 'model_state_dict.pth')

# Load the model's state dict into a new instance of the same model class
inception = ModifiedInceptionV3(num_classes=24).to(device)
inception.load_state_dict(torch.load('model_state_dict.pth'))
inception.eval()  # Set to evaluation mode

# Compare parameters between the original and loaded models
for param_name in model.state_dict():
    original_param = model.state_dict()[param_name]
    loaded_param = inception.state_dict()[param_name]
    if torch.equal(original_param, loaded_param):
        print(f'Parameter {param_name} matches exactly.')
    else:
        print(f'Parameter {param_name} does not match.')

