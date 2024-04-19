from PIL import Image
import os

# Path to the dataset directory
dataset_path = './origin'

# List all files in the dataset directory
file_list = os.listdir(dataset_path)

# Filter for .tiff files
tiff_files = [f for f in file_list if f.endswith('.tiff') or f.endswith('.tif')]

# Check each TIFF file
for tiff_file in tiff_files:
    file_path = os.path.join(dataset_path, tiff_file)
    try:
        # Attempt to open the image
        img = Image.open(file_path)
        img.verify()  # Check if the file is corrupted
        img.close()   # Close the image
        print(f"{tiff_file}: OK")
    except (IOError, SyntaxError) as e:
        # Handle corrupted or unreadable files
        print(f"{tiff_file}: Error - {e}")
