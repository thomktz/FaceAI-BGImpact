import os
import kaggle
from faceai_bgimpact.data_processing.paths import (
    data_folder,
    ffhq_raw_kaggle_path,
    ffhq_grey_kaggle_path,
    ffhq_blur_kaggle_path,
    raw_folder_name,
    original_raw_folder_name,
)


def download_all_ffhq(raw=True, grey=True, blur=True):
    """
    Download raw FFHQ from Kaggle and rename the folder to ffhq_raw.

    Then, download our modified versions of the dataset.

    Parameters:
    ----------
    raw : bool
        Whether to download the raw FFHQ dataset.
    grey : bool
        Whether to download the grey FFHQ dataset.
    blur : bool
        Whether to download the blurred FFHQ dataset.
    """
    # Download the raw FFHQ dataset
    if raw:
        kaggle.api.dataset_download_files(ffhq_raw_kaggle_path, path=data_folder, unzip=True, quiet=False)

        # Check if the original folder exists and rename it
        if os.path.exists(original_raw_folder_name):
            os.rename(original_raw_folder_name, raw_folder_name)
            print(f"Folder renamed from {original_raw_folder_name} to {raw_folder_name}")
        else:
            raise ValueError(f"The folder {original_raw_folder_name} does not exist.")

    # Download the modified versions of the dataset
    if grey:
        kaggle.api.dataset_download_files(ffhq_grey_kaggle_path, path=data_folder, unzip=True, quiet=False)
    if blur:
        kaggle.api.dataset_download_files(ffhq_blur_kaggle_path, path=data_folder, unzip=True, quiet=False)


if __name__ == "__main__":
    download_all_ffhq()
