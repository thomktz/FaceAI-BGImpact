import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from zipfile import ZipFile, ZIP_DEFLATED
from faceai_bgimpact.data_processing.paths import raw_folder_name, mask_folder_name, blur_folder_name, grey_folder_name

# Ensure the output directories exist
Path(blur_folder_name).mkdir(parents=True, exist_ok=True)
Path(grey_folder_name).mkdir(parents=True, exist_ok=True)


def apply_blur(raw_image, mask):
    """Apply a Gaussian blur to the image."""
    # Blur the entire image
    blurred_image = cv2.GaussianBlur(raw_image, (31, 31), 0)
    # Blend the raw image and the blurred image using the mask
    return raw_image * mask[:, :, None] + blurred_image * (1 - mask[:, :, None])


def apply_grey(raw_image, mask):
    """Apply a grey background to the image."""
    # Create a solid grey image
    grey_background = np.full_like(raw_image, (128, 128, 128))
    # Blend the raw image and the grey image using the mask
    return raw_image * mask[:, :, None] + grey_background * (1 - mask[:, :, None])


def create_blurred_and_grey_images():
    """Create blurred and grey images."""
    files = list(Path(mask_folder_name).glob("*.png"))
    for mask_file in tqdm(files, desc="Applying masks"):
        # Corresponding raw image path
        raw_image_path = Path(raw_folder_name) / mask_file.name
        # Load the original image and the mask
        raw_image = cv2.imread(str(raw_image_path))
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED) / 255.0  # Load mask as float and normalize

        # Apply the mask to create blurred and grey background images
        blurred_image = apply_blur(raw_image, mask)
        grey_image = apply_grey(raw_image, mask)

        # Save the processed images
        blurred_image_path = Path(blur_folder_name) / mask_file.name
        grey_image_path = Path(grey_folder_name) / mask_file.name
        cv2.imwrite(str(blurred_image_path), blurred_image)
        cv2.imwrite(str(grey_image_path), grey_image)


def compress_folder(folder_name):
    """Compress a folder."""
    # Create a zip file
    zip_file_name = f"{folder_name}.zip"
    zip_file = ZipFile(zip_file_name, "w", ZIP_DEFLATED)

    # Add all files in the folder to the zip file
    for file in tqdm(Path(folder_name).glob("*"), desc=f"Compressing {folder_name}"):
        zip_file.write(file, file.name)

    # Close the zip file
    zip_file.close()


if __name__ == "__main__":
    create_blurred_and_grey_images()
