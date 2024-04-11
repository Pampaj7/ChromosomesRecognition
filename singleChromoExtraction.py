import cv2
import os
import annotationExtraction as ae
from tqdm import tqdm
def single_chromo_extractor(image_path, annotations, output_dir, image_file):
    # Check if the image file exists and is valid
    if not os.path.exists(image_path) or not os.path.isfile(image_path):
        print(f"Image file does not exist: {image_path}")
        return

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read the image file or the file may be corrupted: {image_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Draw a rectangle around each chromosome
    for idx, ann in enumerate(annotations):
        bndbox = ann['bndbox']
        xmin = int(bndbox['xmin'])
        ymin = int(bndbox['ymin'])
        xmax = int(bndbox['xmax'])
        ymax = int(bndbox['ymax'])

        cropped_image = image[ymin:ymax, xmin:xmax]
        if cropped_image.size == 0:  # some images have empty annotations -- dumb annotators
            print(f"The cropped image is empty for annotation {idx} in image {image_file}. Check your annotations.")
            continue

        # Construct a unique filename for each cropped chromosome
        filename = os.path.splitext(image_file)[0]  # Extract filename without extension
        cropped_image_path = os.path.join(output_dir, f"{filename}_chromosome_{idx}.jpg")
        cv2.imwrite(cropped_image_path, cropped_image)


def complete_extractor(image_dir, annotations_dir, output_dir):
    # Iterate over all the images in the JPEG directory
    for image_file in tqdm(os.listdir(image_dir), desc="Extracting chromosomes"):
        # Check if the corresponding XML file exists
        base_filename, _ = os.path.splitext(image_file)
        xml_file = f"{base_filename}.xml"
        xml_path = os.path.join(annotations_dir, xml_file)

        if os.path.isfile(xml_path):
            annotations = ae.extract_annotations_from_xml(xml_path)
            image_path = os.path.join(image_dir, image_file)
            single_chromo_extractor(image_path, annotations, output_dir, image_file)
        else:
            print(f"No annotation for image {image_file}")



