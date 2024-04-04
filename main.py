import cv2
import matplotlib.pyplot as plt
import os

# Define the directory containing the dataset images
path0 = '/Users/pampaj/Downloads/Data/24_chromosomes_object/JEPG/'

# Load an image from the specified directory
image_path = os.path.join(path0, '1101471.jpg')  # Replace 'example_image.jpg' with the actual filename
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is not None:
    # Display the image using Matplotlib
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
else:
    print("Failed to load the image.")
