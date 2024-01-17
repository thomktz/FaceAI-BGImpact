import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from faceai_bgimpact.data_processing.paths import data_folder


class FFHQDataset(Dataset):
    """FFHQ dataset that returns blended images at two resolutions."""

    def __init__(self, root_dir, resolution, alpha=1.0):
        self.root_dir = root_dir
        try:
            self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
        except FileNotFoundError:
            raise FileNotFoundError("Please download the data first, using the download-all-ffhq command.")
        self.resolution = resolution
        self.alpha = alpha
        self._update_transforms()

    def _update_transforms(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.resolution, self.resolution), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.low_res_transform = transforms.Compose(
            [
                transforms.Resize((self.resolution // 2, self.resolution // 2), antialias=True),
                transforms.Resize((self.resolution, self.resolution), antialias=True),  # Scale back up
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def update_alpha(self, new_alpha):
        """Update the alpha value for progressive growing."""
        self.alpha = new_alpha

    def update_resolution(self, new_resolution):
        """Update the resolution of the images."""
        self.resolution = new_resolution
        self._update_transforms()

    def __getitem__(self, idx):
        """Get the blended image at the specified index."""
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(image_path)

        high_res_image = self.transform(image)
        low_res_image = self.low_res_transform(image)

        blended_image = self.alpha * high_res_image + (1 - self.alpha) * low_res_image
        return blended_image

    def __len__(self):
        """Length of the dataset."""
        return len(self.image_files)


def get_dataloader(dataset_name, batch_size, shuffle=True, resolution=128, alpha=1.0, other_data_folder=None):
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
    # Load the dataset with blended images
    if other_data_folder is None:
        root_dir = f"{data_folder}/{dataset_name}"
    else:
        root_dir = f"{other_data_folder}/{dataset_name}"
    dataset = FFHQDataset(root_dir=root_dir, resolution=resolution, alpha=alpha)

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count())

    return dataset, loader


inv_normalize = transforms.Compose(
    [
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[2.0, 2.0, 2.0]),
        transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1.0, 1.0, 1.0]),
    ]
)


def denormalize_image(tensor):
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
