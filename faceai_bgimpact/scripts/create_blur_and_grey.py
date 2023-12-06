from faceai_bgimpact.data_processing.create_blur_and_grey import create_blurred_and_grey_images, compress_folder
from faceai_bgimpact.data_processing.paths import blur_folder_name, grey_folder_name


def create_blur_and_grey(zip_: bool = True):
    """Create blurred and grey images."""
    create_blurred_and_grey_images()
    if zip_:
        print("Compressing blurred and grey folders...")
        compress_folder(blur_folder_name)
        compress_folder(grey_folder_name)
