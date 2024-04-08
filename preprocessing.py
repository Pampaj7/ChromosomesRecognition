import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_images(original_image, preprocessed):
    """Display the original and preprocessed images side by side."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("First Image")

    plt.subplot(1, 2, 2)
    plt.imshow(preprocessed, cmap='gray')
    plt.title('Preprocessed Image')
    plt.show()


def preprocess_image(image_path):
    # Read the image in grayscale
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Denoising
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 10, 7, 21)  # he rocks

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced_image = clahe.apply(denoised_image)  # he rocks
    """ # can be usefull for resizing, need to check in the cnn
    image = contrast_enhanced_image

    # Calculate the ratio of the target dimensions
    height_ratio, width_ratio = 224 / image.shape[0], 224 / image.shape[1]
    new_ratio = min(height_ratio, width_ratio)

    # Calculate new dimensions
    new_height, new_width = int(image.shape[0] * new_ratio), int(image.shape[1] * new_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Calculate padding
    delta_width = 224 - new_width
    delta_height = 224 - new_height
    top, bottom = delta_height // 2, delta_height - (delta_height // 2)
    left, right = delta_width // 2, delta_width - (delta_width // 2)

    # Apply padding
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    """
    return contrast_enhanced_image


def clean_image(image):

    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    contour_image = np.copy(image)
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 1)  # draw in blue with thickness 1

    return binary_image, contours, contour_image


def separate_overlapping_chromosomes(binary_image):
    # Convert binary image to 8-bit format if not already
    binary_image = np.uint8(binary_image)

    # Invert the image so the background is 0.
    binary_image = 255 - binary_image

    # Perform distance transform
    distance_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 0)
    # Normalize the distance transform for visualization
    normalized_dist_transform = cv2.normalize(distance_transform, None, 255, 0, cv2.NORM_MINMAX)

    # Threshold the distance transform
    _, dist_transform_thresh = cv2.threshold(normalized_dist_transform, 0.1 * normalized_dist_transform.max(), 255, 0)
    dist_transform_thresh = np.uint8(dist_transform_thresh)

    # Determine markers for the Watershed algorithm
    markers = cv2.connectedComponents(dist_transform_thresh)[1]

    # Apply Watershed algorithm
    markers = cv2.watershed(cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR), markers)

    # Generate the segmented image
    segmented_image = np.zeros(binary_image.shape, dtype=np.uint8)
    segmented_image[markers == -1] = 255  # -1 denotes the watershed ridge

    return segmented_image


def remove_small_objects(binary_image, min_area):
    # Find all connected components (aka blobs in the image)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # Create a mask where we will fill in the desired regions
    mask = np.zeros_like(binary_image)

    # Loop through all found components
    for label in range(1, num_labels):  # Start from 1 to ignore the background
        area = stats[label, cv2.CC_STAT_AREA]

        # If the component has sufficient area, keep it
        if area >= min_area:
            mask[labels == label] = 255

    return mask

datasetPath = 'Dataset/Data/24_chromosomes_object/cropped_chromosomes/'
annotationsPath = 'Dataset/Data/24_chromosomes_object/annotations/'
image_path = (datasetPath + "chromosome_3.jpg")
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

preprocessed_image = preprocess_image(image_path)

binaryImage, contours, contoursimage = clean_image(preprocessed_image)  # TODO non fa una sega

segmented_image = separate_overlapping_chromosomes(contoursimage)

noobj = remove_small_objects(binaryImage, 400)

display_images(original_image, noobj)

display_images(original_image, segmented_image)

display_images(contoursimage, binaryImage)

display_images(original_image, preprocessed_image)
