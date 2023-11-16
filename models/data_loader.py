import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch

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

def get_dataloader(dataset_name, batch_size, shuffle=True):
    """
    Create a DataLoader for the specified FFHQ dataset.
    
    Parameters
    ----------
    dataset_name : str
        The name of the dataset to load.
    batch_size : int
        The batch size to use for the DataLoader.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize the image using ImageNet statistics
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the dataset
    root_dir = f"data_processing/{dataset_name}"
    dataset = FFHQDataset(root_dir=root_dir, transform=transform)

    # Create DataLoaders
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
    