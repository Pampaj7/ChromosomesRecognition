import cv2
from PIL import Image
import torch
from torchvision.models import Inception_V3_Weights
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
from ModifiedModels import *

# List of image filenames
filenames = [
"DataNew/1/1_1_56131_115966_1341593.tiff"

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
    if (model == 0):
        return ModifiedInceptionV3(24)
    elif (model == 1):
        return ModifiedInceptionV3Paper(24)


model_name = "models/V3ModInception.pt"
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
    image = cv2.imread(filename)
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #image = Image.open(filename).convert('RGB')
    #input_image = transform(image).unsqueeze(0)

    #input_image = input_image.to(device)
    input_image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        outputs = inception(input_image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        # Get the class index with the highest probability
        predicted = torch.argmax(probabilities).item()
        correct_class = index_to_class[predicted]

    print(f"Predicted class for {filename}: {correct_class}")
