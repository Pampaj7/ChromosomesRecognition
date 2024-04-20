from PIL import Image
import torch
from torchvision.models import Inception_V3_Weights
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn

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
    "dataset/DataGood/ChromoClassified/0/7501562828258.235663.0.tiff",
    "dataset/DataGood/ChromoClassified/0/16531562828258.3343055.0.tiff",
    "dataset/DataGood/ChromoClassified/23/3241562828258.549619.23.tiff",
    "dataset/DataGood/ChromoClassified/23/41861562828257.9127784.23.tiff",
    "dataset/DataGood/ChromoClassified/23/80611562828258.843462.23.tiff",
    "dataset/DataGood/ChromoClassified/15/10761562828258.4776962.15.tiff",
    "dataset/DataGood/ChromoClassified/14/12691562828259.3142326.14.tiff",
    "Dataset/Data/24_chromosomes_object/preprocessed_images/chromo0/103111_chromosome_0.jpg",
    "Dataset/Data/24_chromosomes_object/cropped_chromosomes/103064_chromosome_3.jpg"
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

class_dict = {'0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7, '16': 8, '17': 9, '18': 10,
              '19': 11, '2': 12, '20': 13, '21': 14, '22': 15, '23': 16, '3': 17, '4': 18, '5': 19, '6': 20, '7': 21,
              '8': 22, '9': 23}
index_to_class = {v: k for k, v in class_dict.items()}


class ModifiedInceptionV3(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedInceptionV3, self).__init__()
        self.inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception.Conv2d_1a_3x3.conv = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.inception.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.inception(x)


model_name = "models/V3ModInception.pt"
inception = ModifiedInceptionV3(num_classes=24).to(device)

# Load the TorchScript model
inception.load_state_dict(torch.load(model_name))
inception.eval()

# Inference transformations, ensuring grayscale images are treated correctly
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
])

# Check if GPU is available and move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
inception.to(device)

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
