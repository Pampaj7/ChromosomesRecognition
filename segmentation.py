import cv2
import matplotlib.pyplot as plt
import preprocessing as pp
import numpy as np


def segment_chromosomes(preprocessed_binary_image, original_image_path):
    # Assume preprocessed_binary_image is already a binary image with chromosomes as foreground

    # Invert the image so the chromosomes are white
    inverted_image = cv2.bitwise_not(preprocessed_binary_image)

    # Compute the distance transform
    dist_transform = cv2.distanceTransform(inverted_image, cv2.DIST_L2, 5)

    # Normalize the distance image for display
    normalized_dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)

    # Threshold to get the peaks
    _, dist_thresh = cv2.threshold(normalized_dist_transform, 0.5 * normalized_dist_transform.max(), 255, 0)

    # Dilate a bit to make the chromosome markers more robust
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(dist_thresh, kernel, iterations=1)

    # Find sure foreground area by dilating the thresholded distance transform
    sure_fg = np.uint8(dilated)

    # Find unknown region by subtracting the sure foreground from the binary image
    sure_fg = cv2.erode(sure_fg, kernel, iterations=2)
    unknown = cv2.subtract(inverted_image, sure_fg)

    # Label the sure foreground
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Load the original image for watershed
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        raise IOError(f"Cannot load the original image from the path: {original_image_path}")

    # Apply the Watershed algorithm to segment the chromosomes
    markers = cv2.watershed(original_image, markers)

    # Generate an output image to superimpose on the original
    image_with_boundaries = original_image.copy()
    image_with_boundaries[markers == -1] = [255, 0, 0]

    return image_with_boundaries


datasetPath = 'Dataset/Data/24_chromosomes_object/JEPG/'
original_image_path = datasetPath + "103064.jpg"
preprocessed_binary_image = pp.preprocess_image(original_image_path)
original_image = cv2.imread(original_image_path)

segmented_image = segment_chromosomes(preprocessed_binary_image, original_image_path)

# Convert the segmented image from BGR to RGB for displaying
segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

# Display the image using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(segmented_image_rgb)
plt.title('Segmented Chromosomes')
plt.axis('off')  # Hide the axes
plt.show()
