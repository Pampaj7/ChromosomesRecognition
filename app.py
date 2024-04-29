from flask import *
from flask_cors import CORS
from PIL import Image
import torch
from torchvision.models import Inception_V3_Weights
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn

app = Flask(__name__)
CORS(app)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModifiedInceptionV3(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedInceptionV3, self).__init__()
        self.inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception.Conv2d_1a_3x3.conv = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.inception.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.inception(x)


model_name = "V3ModInception.pt"
inception = ModifiedInceptionV3(num_classes=24).to(device)

# Load the TorchScript model
map_location = 'cpu' if not torch.cuda.is_available() else None
inception.load_state_dict(torch.load("models/" + model_name, map_location=map_location))
class_dict = {'0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7, '16': 8, '17': 9, '18': 10,
              '19': 11, '2': 12, '20': 13, '21': 14, '22': 15, '23': 16, '3': 17, '4': 18, '5': 19, '6': 20, '7': 21,
              '8': 22, '9': 23}
index_to_class = {v: k for k, v in class_dict.items()}


@app.route('/getChromosome/', methods=['POST'])
def getChromosome():
    inception.eval()

    # Inference transformations, ensuring grayscale images are treated correctly
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
    ])
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Read the image file from the request
    img = request.files['image']

    image = Image.open(img).convert('RGB')
    input_image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = inception(input_image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        # Get the class index with the highest probability
        predicted = torch.argmax(probabilities).item()
        correct_class = index_to_class[predicted]
    return jsonify({'predictions': correct_class})


if __name__ == '__main__':
    app.run(port=8888, debug=True)
