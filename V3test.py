from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
from torchvision.models import Inception_V3_Weights

num_classes = 24

# Load the saved model
inception = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
# Assuming num_classes is known
inception.fc = nn.Linear(inception.fc.in_features, num_classes)
inception.load_state_dict(torch.load('Chromo_model.pth'))
inception.eval()  # Set the model to evaluation mode

# Load a single image
image_path = 'dataset/DataGood/ChromoClassified/2/7991562828257.811051.2.tiff'
image = Image.open(image_path)

# Convert grayscale image to RGB
image_rgb = image.convert('RGB')

# Apply the same transformations as in training
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# Preprocess the image
input_image = transform(image_rgb).unsqueeze(0)

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

# Move the model to the same device as the input tensor
inception.to(device)
input_image = input_image.to(device)

# Perform inference
with torch.no_grad():
    output = inception(input_image)
    _, predicted_class = torch.max(output, 1)

print("Predicted class:", predicted_class.item())
