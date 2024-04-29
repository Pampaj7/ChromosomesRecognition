# Function to parse and extract annotation details from XML file
# XML is hierarchically structured data, so we need to parse it to extract the information we need :(
import xml.etree.ElementTree as ET
import os
from shutil import copy2
from PIL import Image


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


def sort_images_by_class(source_directory, target_directory):
    # Ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)

    # Iterate through all files in the source directory
    for filename in os.listdir(source_directory):
        if filename.endswith(".tiff"):
            # Extract the class from the filename
            class_label = filename.split('.')[-2]
            class_dir = os.path.join(target_directory, class_label)

            # Ensure the class directory exists
            os.makedirs(class_dir, exist_ok=True)

            # Construct the full file paths
            file_path = os.path.join(source_directory, filename)
            target_path = os.path.join(class_dir, filename)

            # Copy the file to the new directory
            copy2(file_path, target_path)
            print(f"Copied {filename} to {class_dir}/")


# Example usage
source_dir = 'dataset/DataGood/origin/'  # Change this to your source directory
target_dir = 'dataset/DataGood/ChromoClassified'  # Change this to your target directory


# sort_images_by_class(source_dir, target_dir)

def convert_tiff_to_jpeg(source_directory="dataset/DataGood/origin", target_directory="dataset/DataGood/origin_jpeg"):
    # Ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)

    # Iterate over all files in the source directory
    for filename in os.listdir(source_directory):
        if filename.endswith(".tiff"):
            # Open the image file
            with Image.open(os.path.join(source_directory, filename)) as img:
                # Define the new filename
                new_filename = filename.replace(".tiff", ".jpeg")

                # Set the target path for the JPEG
                target_path = os.path.join(target_directory, new_filename)

                # Convert the image to RGB mode if not already (JPEG doesn't support alpha channel as in RGBA)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Save the image in JPEG format
                img.save(target_path, "JPEG", quality=85)  # You can adjust the quality level

                print(f"Converted {filename} to {new_filename} and saved to {target_directory}")

# convert_tiff_to_jpeg()
