import torch
import torch.nn as nn
import torch.nn.init as init

def weights_init(std=0.02):
    def f(m):
        """
        Initialize the weights of a neural network module.
        
        Parameters:
        ----------
        m : nn.Module
            A module of the neural network.
        """
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            # Apply weight initialization to convolutional and linear layers
            init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            # Apply weight initialization to batch normalization layers
            if m.weight is not None:
                init.normal_(m.weight.data, 1.0, std)
            if m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
    return f

def pairwise_euclidean_distance(batch):
    """
    Compute pairwise Euclidean distance for a batch of images.

    Parameters:
    -----------
    batch: torch.Tensor
        A batch of images. Shape: (batch_size, channels, height, width)

    Returns:
    --------
    distances: torch.Tensor
        Pairwise distances of all images in the batch.
    """
    with torch.no_grad():
        # Flatten the images
        flattened = batch.view(batch.size(0), -1)

        # Compute pairwise distance
        dist_matrix = torch.cdist(flattened, flattened, p=2)

        # Optionally, you can return the mean distance for a simple metric
        mean_dist = dist_matrix.mean()

        return mean_dist