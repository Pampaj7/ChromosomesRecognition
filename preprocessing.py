import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


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

    # Thresholding or edge detection
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #edged_image = cv2.Canny(blurred_image, 100, 200)

    return binary_image


def remove_unwanted_objects(binary_image, min_area, max_area):
    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Visualize contours
    image_contours = cv2.cvtColor(binary_image.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 10)  # Draw all contours
    plt.figure(figsize=(6, 6))
    plt.title("All Contours")
    plt.imshow(image_contours)
    plt.show()

    # Create a mask where we will draw the objects we want to keep
    mask = np.zeros_like(binary_image)

    # Filter contours by area and fill the ones we want to keep in the mask ##TODO not working
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            cv2.drawContours(mask, [cnt], -1, (255), thickness=cv2.FILLED)

    # Visualize filtered contours
    image_filtered_contours = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
    plt.figure(figsize=(6, 6))
    plt.title("Filtered Contours")
    plt.imshow(image_filtered_contours)
    plt.show()

    # The mask now contains the objects we want to keep
    return mask


def display_images(original, preprocessed):
    """Display the original and preprocessed images side by side."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)  # Convert BGR to RGB
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(preprocessed, cmap='gray')
    plt.title('Preprocessed Image')
    plt.show()


datasetPath = 'Dataset/Data/24_chromosomes_object/JEPG/'
annotationsPath = 'Dataset/Data/24_chromosomes_object/annotations/'
image_size = (224, 224)
preprocessed_images = []
labels = []
image_path = (datasetPath + "103064.jpg")
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for displaying
preprocessed_image = preprocess_image(image_path)

cleaned_image = remove_unwanted_objects(preprocessed_image, 200, 10000)

display_images(original_image, preprocessed_image)
