import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import random_split

class FFHQDataset(Dataset):
    """
    Dataloader for the FFHQ dataset.
    
    Parameters
    ----------
    root_dir : str
        The root directory of the dataset.
    transform : torchvision.transforms
        The transformations to apply to the images.
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image

def get_dataloader(dataset_name, batch_size):
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images to [-1, 1]
    ])

    # Load the dataset
    root_dir = f"data_processing/{dataset_name}"
    dataset = FFHQDataset(root_dir=root_dir, transform=transform)

    # Create DataLoaders
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader