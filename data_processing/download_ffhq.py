import kaggle

# Paths
dataset_path = "greatgamedota/ffhq-face-data-set"
data_folder = "data_processing/"
original_folder = data_folder + "thumbnails128x128"
new_folder = data_folder + "ffhq_raw"

def download_ffhq():
    """Download raw FFHQ from Kaggle and rename the folder to ffhq_raw."""
    
    print("Make sure to run this script from the root, and not from data_processing!")
    kaggle.api.dataset_download_files(dataset_path, path=data_folder, unzip=True, quiet=False)

    # Check if the original folder exists and rename it
    if os.path.exists(original_folder):
        os.rename(original_folder, new_folder)
        print(f"Folder renamed from {original_folder} to {new_folder}")
    else:
        raise ValueError(f"The folder {original_folder} does not exist.")


if __name__ == "__main__":
    download_ffhq()
    
