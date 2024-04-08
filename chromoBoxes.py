import cv2
import preprocessing as pp
import os
import matplotlib.pyplot as plt
import annotationExtraction as ae

def draw_boxes_around_chromosomes(image_path, annotations):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    output_dir = "dataset/Data/24_chromosomes_object/cropped_chromosomes"
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

        # Draw a rectangle around the chromosome
        #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Construct a unique filename for each cropped chromosome
        cropped_image_path = os.path.join(output_dir, f"chromosome_{idx}.jpg")
        cv2.imwrite(cropped_image_path, cropped_image)

    return image


annotations = ae.extract_annotations_from_xml('dataset/Data/24_chromosomes_object/annotations/103064.xml')
image_with_boxes = draw_boxes_around_chromosomes('Dataset/Data/24_chromosomes_object/JEPG/103064.jpg', annotations)

plt.imshow(image_with_boxes)
plt.show()
