from PIL import Image
import torch
from torchvision.models import Inception_V3_Weights
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
from ModifiedModels import *

# List of image filenames
filenames = [
    "origin/01562828258.8049057.5.tiff", "origin/21562828259.126213.3.tiff", "origin/431562828258.5368197.3.tiff", "origin/691562828258.3360758.5.tiff", "origin/731562828258.3894138.7.tiff",
    "origin/1351562828258.40041.5.tiff", "origin/311562828259.323763.2.tiff", "origin/2461562828257.9331849.20.tiff", "origin/4071562828258.0701625.14.tiff",
    "origin/4671562828258.9595804.3.tiff", "origin/6421562828258.7006133.8.tiff", "origin/9011562828258.2214892.9.tiff", "origin/14401562828258.2325635.9.tiff",
    
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_dict = {'0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7, '16': 8, '17': 9, '18': 10,
              '19': 11, '2': 12, '20': 13, '21': 14, '22': 15, '23': 16, '3': 17, '4': 18, '5': 19, '6': 20, '7': 21,
              '8': 22, '9': 23}
index_to_class = {v: k for k, v in class_dict.items()}

def chooseModel(model):
    #
    # 0 = ModifiedInceptionV3
    # 1 = ModifiedInceptionV3Paper
    if( model == 0):
        return ModifiedInceptionV3(24)
    elif( model == 1):
        return ModifiedInceptionV3Paper(24)

model_name = "ChromosomesRecognition/models/V3ModInception.pt"
inception = chooseModel(0).to(device)

# Load the TorchScript model
# Set map_location to 'cpu' if CUDA isn't available, otherwise use the default which is to load on the current CUDA device
map_location = 'cpu' if not torch.cuda.is_available() else None
inception.load_state_dict(torch.load(model_name, map_location=map_location))
inception.eval()

# Inference transformations, ensuring grayscale images are treated correctly
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
])

# Check if GPU is available and move model to GPU if available
path = "dataset/DataGood/ChromoClassified/18/"

# Iterate over the list of filenames
for filename in filenames:
    # Load and preprocess the image
    image = Image.open(filename).convert('RGB')
    input_image = transform(image).unsqueeze(0)

    input_image = input_image.to(device)
    with torch.no_grad():
        outputs = inception(input_image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        # Get the class index with the highest probability
        predicted = torch.argmax(probabilities).item()
        correct_class = index_to_class[predicted]

    print(f"Predicted class for {filename}: {correct_class}")
