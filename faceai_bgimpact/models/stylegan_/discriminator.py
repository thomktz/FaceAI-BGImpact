import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from faceai_bgimpact.models.stylegan_.utils import WSConv2d, WSLinear, BlurLayer

class DiscriminatorBlock(nn.Module):
    """
    Discriminator Block for a specific resolution with downscaling.
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = WSConv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = WSConv2d(in_channels, out_channels, 3, 1, 1)
        self.blur = BlurLayer()
        self.activation = nn.LeakyReLU(0.2)
        self.downsample = nn.AvgPool2d(2)
        
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.blur(x)
        x = self.activation(self.conv2(x))
        x = self.downsample(x)
        return x

class LastDiscriminatorBlock(nn.Module):
    """
    Last discriminator block
    """
    
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = WSConv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = WSConv2d(in_channels, in_channels, 4, 1, 0)
        self.conv3 = WSConv2d(in_channels, 1, 1, 1, 0, gain=1)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)
        return x
    
class Discriminator(nn.Module):
    """
    Discriminator Synthesis Network.
    """
    
    def __init__(self):
        super().__init__()
        
        # Assuming the highest resolution is 128x128, we define the from_rgb_layers
        self.from_rgb_layers = nn.ModuleList([
            WSConv2d(3, 16, 1, 1, 0),   # For 128x128 resolution
            WSConv2d(3, 32, 1, 1, 0),   # For 64x64 resolution
            WSConv2d(3, 64, 1, 1, 0),   # For 32x32 resolution
            WSConv2d(3, 128, 1, 1, 0),  # For 16x16 resolution
            WSConv2d(3, 256, 1, 1, 0),  # For 8x8 resolution
            WSConv2d(3, 256, 1, 1, 0)   # For 4x4 resolution
        ])
        
        # Assuming the highest resolution is 128x128, we define the downscale_blocks
        self.downscale_blocks = nn.ModuleList([
            DiscriminatorBlock(16, 32),    # For 128x128 to 64x64
            DiscriminatorBlock(32, 64),    # For 64x64 to 32x32
            DiscriminatorBlock(64, 128),   # For 32x32 to 16x16
            DiscriminatorBlock(128, 256),  # For 16x16 to 8x8
            DiscriminatorBlock(256, 256),  # For 8x8 to 4x4
            LastDiscriminatorBlock(256)    # For 4x4 to 1
        ])
        
        self.len_layers = len(self.downscale_blocks)
        
    def forward(self, img, current_level, alpha):
        """
        Forward pass with progressive growing.
        
        Parameters:
        ----------
        img : torch.Tensor
            Input image tensor.
        current_level : int
            Current resolution level for progressive growing (e.g., 0 for 4x4, 1 for 8x8, etc.).
        alpha : float
            Blending factor for progressive growing.

        Returns:
        ----------
        torch.Tensor: Discriminator's output.
        """
        # Downsample the input image to the current resolution level
        first_layer = self.len_layers - current_level - 1
        
        if alpha < 1.0:
            # First, downsample the image to the next resolution level and make it a latent vector
            downsampled_image = F.avg_pool2d(img, 2)
            vector_from_downsampled_image = self.from_rgb_layers[first_layer + 1](downsampled_image)
            
            # Then, make the input image a latent vector and downsample the result
            vector = self.from_rgb_layers[first_layer](img)
            downsampled_vector = self.downscale_blocks[first_layer](vector)
            
            # Blend the two latent vectors
            x = (1 - alpha) * vector_from_downsampled_image + alpha * downsampled_vector
            
        else:
            # Normal processing
            # rgb_img -> latent vector -> downsample
            x = self.from_rgb_layers[first_layer](img)
            x = self.downscale_blocks[first_layer](x)
        
        # Skipping level=first_layer because it is already processed above
        for level in range(first_layer + 1, self.len_layers):
            x = self.downscale_blocks[level](x)
        
        # Flatten the output
        return x.view(x.shape[0], -1)
        
        