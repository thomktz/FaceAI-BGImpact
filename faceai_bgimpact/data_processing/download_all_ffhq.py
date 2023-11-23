import os
import kaggle
from faceai_bgimpact.data_processing.paths import (
    data_folder,
    ffhq_raw_kaggle_path,
    ffhq_grey_kaggle_path,
    ffhq_blur_kaggle_path,
    ffhq_stats_kaggle_path,
    raw_folder_name,
    original_raw_folder_name
)

def download_all_ffhq():
    """
    Download raw FFHQ from Kaggle and rename the folder to ffhq_raw.
    
    Then, download our modified versions of the dataset, as well as FID statistics.
    """
    
    # Download the raw FFHQ dataset
    kaggle.api.dataset_download_files(ffhq_raw_kaggle_path, path=data_folder, unzip=True, quiet=False)

    # Check if the original folder exists and rename it
    if os.path.exists(original_raw_folder_name):
        os.rename(original_raw_folder_name, raw_folder_name)
        print(f"Folder renamed from {original_raw_folder_name} to {raw_folder_name}")
    else:
        raise ValueError(f"The folder {original_raw_folder_name} does not exist.")

    kaggle.api.dataset_download_files(ffhq_grey_kaggle_path, path=data_folder, unzip=True, quiet=False)
    kaggle.api.dataset_download_files(ffhq_blur_kaggle_path, path=data_folder, unzip=True, quiet=False)
    
    # Download the FID statistics
    kaggle.api.dataset_download_files(ffhq_stats_kaggle_path, path=data_folder, unzip=True, quiet=False)

if __name__ == "__main__":
    download_all_ffhq()
    
