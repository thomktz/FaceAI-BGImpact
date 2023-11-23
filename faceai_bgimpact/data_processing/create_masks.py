from pathlib import Path
from tqdm import tqdm
from rembg import remove, new_session
from faceai_bgimpact.data_processing.paths import raw_folder_name, mask_folder_name

session = new_session()

def create_masks():
    """Create cutout masks for the images."""
    
    # Ensure the mask_folder_name directory exists
    mask_folder_path = Path(mask_folder_name)
    mask_folder_path.mkdir(parents=True, exist_ok=True)
    
    # Get all the png files from raw_folder_name
    files = list(Path(raw_folder_name).glob("*.png"))

    # Initialize tqdm progress bar
    for file in tqdm(files, desc="Creating masks"):
        input_path = file
        output_path = mask_folder_path / (file.stem + ".png")

        with open(input_path, "rb") as i:
            with open(output_path, "wb") as o:
                input_data = i.read()
                output_data = remove(input_data, session=session, only_mask=True)
                o.write(output_data)

if __name__ == "__main__":
    create_masks()
