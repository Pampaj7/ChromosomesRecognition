from PIL import Image

def extract_tiff_metadata(file_path):
    with Image.open(file_path) as img:
        meta_data = img.info
        print("Metadata in TIFF file:")
        for key, value in meta_data.items():
            print(f"{key}: {value}")

# Replace 'path_to_your_tiff_file.tiff' with the path to your TIFF file
extract_tiff_metadata('19701562828258.415899.6.tiff')
