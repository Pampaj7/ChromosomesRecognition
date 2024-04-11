import singleChromoExtraction as SCE
import preprocessing as pp

# Directories
image_dir = 'Dataset/Data/24_chromosomes_object/JEPG'
annotations_dir = 'Dataset/Data/24_chromosomes_object/annotations'
cropped_dir = "Dataset/Data/24_chromosomes_object/cropped_chromosomes"
preprocessed_dir = "Dataset/Data/24_chromosomes_object/preprocessed_images"

# Run the complete extraction
# uncomment to redo
# SCE.complete_extractor(image_dir, annotations_dir, cropped_dir)

# Run the preprocessing
# uncomment to redo
pp.preprocess_directory(cropped_dir, preprocessed_dir)
