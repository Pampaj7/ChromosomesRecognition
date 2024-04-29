import preprocessing as pp
import matplotlib.pyplot as plt
import numpy as np
import cv2


# carefull, an object like a U or a C will have a smaller caliper distance, but it is not the length of the chromosome
def find_max_caliper_distance(contour):
    max_dist = 0
    point_one = point_two = (0, 0)

    for i in range(len(contour)):
        for j in range(i + 1, len(contour)):
            dist = np.linalg.norm(contour[i] - contour[j])
            if dist > max_dist:
                max_dist = dist
                point_one = tuple(contour[i])
                point_two = tuple(contour[j])

    return max_dist, point_one, point_two


def skeletonize(image):
    # Initialize the skeletonized image
    skeleton = np.zeros(image.shape, np.uint8)

    # Get a cross-shaped structuring element for morphological operations
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Repeat until no more pixels can be removed
    while True:
        # Erode the image
        eroded = cv2.erode(image, element)
        # Use dilation to reconstruct the image from the eroded version
        temp = cv2.dilate(eroded, element)
        # Subtract the reconstructed image from the original image to get the edges
        temp = cv2.subtract(image, temp)
        # Or the edges (temp) with the skeleton to add the new edges found in this iteration
        skeleton = cv2.bitwise_or(skeleton, temp)
        # Update the image to the eroded version
        image = eroded.copy()

        # If the image is completely eroded away, the skeletonization is done
        if cv2.countNonZero(image) == 0:
            break

    return skeleton


def calculate_skeleton_length(pruned_skeleton):
    points = np.argwhere(pruned_skeleton)  # Extract the coordinates of the white pixels
    length = 0
    for i in range(len(points) - 1):
        length += np.linalg.norm(points[i + 1] - points[i])
    return length


def thinning(skeleton):
    """ Apply morphological thinning to reduce to a single pixel width """
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros(skeleton.shape, np.uint8)
    done = False

    while not done:
        eroded = cv2.erode(skeleton, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(skeleton, temp)
        skel = cv2.bitwise_or(skel, temp)
        skeleton = eroded.copy()

        done = (cv2.countNonZero(skeleton) == 0)

    return skel


def pruning(skeleton, prune_size):
    """ Remove small branches from the skeleton """
    # Find all endpoints of the skeleton
    # An endpoint is defined as a pixel with only one neighbor
    filtered_skeleton = skeleton.copy()
    for i in range(prune_size):
        # A pixel with only one neighbor will be blacked out
        filtered_skeleton = cv2.ximgproc.thinning(filtered_skeleton, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return filtered_skeleton


# Load and display the resized image
image = pp.noobj
plt.imshow(image, cmap='gray')
plt.title("no object image")
plt.show()

# Extract contours from the binary image
contours = pp.contours_extractor(image)
print("Number of contours found:", len(contours))

# Finding the largest contour
largest_contour = max(contours, key=cv2.contourArea)
length, pt1, pt2 = find_max_caliper_distance(largest_contour.squeeze())

print(f"Estimated length of the chromosome with caliper distance: {length}")
print(f"Between points: {pt1} and {pt2}")

skeleton = skeletonize(image)
thinned_skeleton = thinning(skeleton)
# pruned_skeleton = pruning(thinned_skeleton, 10)

length_of_skeleton = calculate_skeleton_length(thinned_skeleton)
print(f"The length of the pruned skeleton is approximately: {length_of_skeleton} pixels")

# Show the pruned skeleton
plt.imshow(skeleton, cmap='gray')
plt.title("Pruned Skeleton")
plt.show()
