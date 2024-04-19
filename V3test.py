from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import Inception_V3_Weights
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm
import os

#{'0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7, '16': 8, '17': 9, '18': 10, '19': 11, '2': 12, '20': 13, '21': 14, '22': 15, '23': 16, '3': 17, '4': 18, '5': 19, '6': 20, '7': 21, '8': 22, '9': 23}


# List of image filenames
filenames = [
    "dataset/DataGood/ChromoClassified/6/731562828257.7667353.6.tiff",
    "dataset/DataGood/ChromoClassified/0/3451562828258.492685.0.tiff",
    "dataset/DataGood/ChromoClassified/18/3351562828258.227251.18.tiff",
    "dataset/DataGood/ChromoClassified/21/2471562828258.225025.21.tiff",
    "dataset/DataGood/ChromoClassified/20/2461562828257.9331849.20.tiff",
    "dataset/DataGood/ChromoClassified/7/5601562828258.1498268.7.tiff",
    "dataset/DataGood/ChromoClassified/15/121562828259.1321106.15.tiff",
    "dataset/DataGood/ChromoClassified/8/6871562828259.130795.8.tiff",
    "dataset/DataGood/ChromoClassified/3/2301562828258.5252967.3.tiff",
]

model_name = "V3Inception.pt"
inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)

num_ftrs = inception.fc.in_features
# num_classes is the number of chromosome classes
inception.fc = torch.nn.Linear(num_ftrs, 24)
# Load the TorchScript model
inception.load_state_dict(torch.load("models/" + model_name))
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
inception.to(device)
inception.eval()

testing_before = True
if testing_before:
    # Testing phase again
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

    data_dir = 'dataset/DataGood/ChromoClassified'
    full_dataset = ImageFolder(root=data_dir, transform=transform)
    print(full_dataset.class_to_idx)

    train_size = int(0.7 * len(full_dataset))
    validation_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - validation_size
    train_dataset, validation_dataset, test_dataset = random_split(
        full_dataset, [train_size, validation_size, test_size])

    # DataLoader setup
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize lists for storing metrics
    train_losses, train_accuracies = [], []
    validation_losses, validation_accuracies = [], []
    test_losses, test_accuracies = [], []

    best_validation = 0.0
    correct_test, total_test = 0.0, 0
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing Batch'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = inception(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

    print(correct_test / total_test)

# Inference transformations, ensuring grayscale images are treated correctly
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
])

# Check if GPU is available and move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
inception.to(device)

# Iterate over the list of filenames
for filename in filenames:
    # Load and preprocess the image
    image = Image.open(path + filename).convert('RGB')
    input_image = transform(image).unsqueeze(0)

    input_image = input_image.to(device)
    print(input_image.size())
    # Perform inference
    with torch.no_grad():
        outputs = inception(input_image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        # Get the class index with the highest probability
        predicted = torch.argmax(probabilities).item()

    print(f"Predicted class for {filename}: {predicted}")
