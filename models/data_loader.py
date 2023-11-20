import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch

class FFHQDataset(torch.utils.data.Dataset):
    """
    FFHQ dataset that returns blended images at two resolutions.
    """
    def __init__(self, root_dir, resolution, alpha):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
        self.resolution = resolution
        self.alpha = alpha
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.low_res_transform = transforms.Compose([
            transforms.Resize((resolution // 2, resolution // 2), antialias=True),
            transforms.Resize((resolution, resolution), antialias=True),  # Scale back up
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        # Load image from dataset (pseudo code)
        image = Image.open(f"{self.root_dir}/{str(idx).zfill(5)}.png")

        # Transform to high resolution and low resolution
        high_res_image = self.transform(image)
        low_res_image = self.low_res_transform(image)

        # Blend images
        blended_image = self.alpha * high_res_image + (1 - self.alpha) * low_res_image
        return blended_image

    def __len__(self):
        return len(self.image_files)

def get_dataloader(dataset_name, batch_size, shuffle=True, resolution=128, alpha=1.0):
    """
    Create a DataLoader for the specified FFHQ dataset.
    
    Parameters
    ----------
    dataset_name : str
        The name of the dataset to load.
    batch_size : int
        The batch size to use for the DataLoader.
    shuffle : bool
        Whether to shuffle the dataset.
    resolution : int
        The resolution of the images in the dataset.
    alpha : float
        The alpha value for progressive growing.
    """
    print(f"Loading dataset: {dataset_name} with resolution {resolution} and alpha {alpha}")        

    # Load the dataset with blended images
    root_dir = f"data_processing/{dataset_name}"
    dataset = FFHQDataset(root_dir=root_dir, resolution=resolution, alpha=alpha)

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader

inv_normalize = transforms.Compose([
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[2.0, 2.0, 2.0]),
    transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1.0, 1.0, 1.0])
])

def denormalize_imagenet(tensor):
    """
    Reverses the ImageNet normalization applied to images.

    Parameters:
    ----------
    tensor : torch.Tensor
        The normalized tensor.
        
    Returns:
    -------
    torch.Tensor
        The denormalized tensor.
    """
    return inv_normalize(tensor)
    