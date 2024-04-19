from PIL import Image
import torch
from torchvision import transforms

# List of image filenames
filenames = [
    "dataset/DataGood/ChromoClassified/6/731562828257.7667353.6.tiff",
    "dataset/DataGood/ChromoClassified/0/3451562828258.492685.0.tiff",
    "dataset/DataGood/ChromoClassified/18/3351562828258.227251.18.tiff", 
    "dataset/DataGood/ChromoClassified/21/2471562828258.225025.21.tiff", 
    "dataset/DataGood/ChromoClassified/20/2461562828257.9331849.20.tiff", 
    "dataset/DataGood/ChromoClassified/7/5601562828258.1498268.7.tiff", 
    
]

# Load the TorchScript model
inception = torch.load("Chromo_model.pt")
inception.eval()

# Inference transformations, ensuring grayscale images are treated correctly
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize as per the training
    # Convert grayscale and duplicate across 3 channels
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    # Normalize as RGB but all channels are the same
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
])


# Check if GPU is available and move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
inception.to(device)

# Iterate over the list of filenames
for filename in filenames:
    # Load and preprocess the image
    image = Image.open(filename).convert('L')
    input_image = transform(image).unsqueeze(0)
    input_image = input_image.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = inception(input_image)
        # Extract only the main output if InceptionOutputs tuple is returned
        main_output = outputs.logits if hasattr(
            outputs, 'logits') else outputs[0]
        _, predicted_class = torch.max(main_output, 1)

    print(f"Predicted class for {filename}: {predicted_class.item()}")
