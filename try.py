import os

# Directories
image_dir = 'dataset/Data/24_chromosomes_object/JEPG'
annotations_dir = 'dataset/Data/24_chromosomes_object/annotations'

# Get the list of image filenames without the file extension
image_files = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
# Get the list of annotation filenames without the file extension
annotation_files = [os.path.splitext(f)[0] for f in os.listdir(annotations_dir) if os.path.isfile(os.path.join(annotations_dir, f))]

# Find the difference between the two sets
uncorrelated_files = set(image_files) - set(annotation_files)

# Print the filenames that are uncorrelated
if uncorrelated_files:
    print("The following image files are missing corresponding annotation files:")
    for file in uncorrelated_files:
        print(file)
else:
    print("All image files have corresponding annotation files.")