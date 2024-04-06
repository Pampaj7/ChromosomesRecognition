import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# Define the directory containing the dataset images and annotations
datasetPath = 'Dataset/Data/24_chromosomes_object/JEPG/'
annotationsPath = 'Dataset/Data/24_chromosomes_object/annotations/'
image_size = (224, 224)
preprocessed_images = []
labels = []


# Function to parse and extract annotation details from XML file
#XML is hierarchically structured data, so we need to parse it to extract the information we need :(
def extract_annotations(element):
    annotations = []
    for obj in element.findall('object'):
        annotation = {
            'name': obj.find('name').text,
            'pose': obj.find('pose').text,
            'truncated': obj.find('truncated').text,
            'difficult': obj.find('difficult').text,
            'bndbox': {
                'xmin': int(obj.find('bndbox/xmin').text),
                'ymin': int(obj.find('bndbox/ymin').text),
                'xmax': int(obj.find('bndbox/xmax').text),
                'ymax': int(obj.find('bndbox/ymax').text)
            }
        }
        annotations.append(annotation)
    return annotations


def extract_annotations_from_xml(xml_file):
    tree = ET.parse(xml_file)  # crazy shit by gpt
    root = tree.getroot()
    return extract_annotations(root)


def preprocess_image(image_path):
    # Read the image in grayscale
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Noise reduction
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Contrast enhancement
    equalized_image = cv2.equalizeHist(blurred_image)

    # Thresholding or edge detection
    # _, binary_image = cv2.threshold(equalized_image, 128, 255, cv2.THRESH_BINARY)
    edged_image = cv2.Canny(equalized_image, 100, 200)

    return edged_image


image_path = (datasetPath + "103064.jpg")
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for displaying
preprocessed_image = preprocess_image(image_path)

# Display the original and preprocessed images
plt.figure(figsize=(10, 5))

# Plot original image
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.imshow(original_image)
plt.title('Original Image')

# Plot preprocessed image
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.imshow(preprocessed_image, cmap='gray')
plt.title('Preprocessed Image')

plt.show()
