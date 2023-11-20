import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stylegan_.utils import SConv2d

class DiscriminatorBlock(nn.Module):
    """
    Discriminator Block for a specific resolution with downscaling.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, downsample=True):
        super(DiscriminatorBlock, self).__init__()
        self.conv1 = SConv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = SConv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.activation = nn.LeakyReLU(0.2)
        self.downsample = nn.AvgPool2d(2) if downsample else nn.Identity()
        self.scale = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.downsample(x)
        x = self.scale * x
        return x
    
class Discriminator(nn.Module):
    """
    Discriminator Synthesis Network.
    """
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Assuming the highest resolution is 128x128, we define the from_rgb_layers
        self.from_rgb_layers = nn.ModuleList([
            SConv2d(3, 16, 1),  # For 128x128 resolution
            SConv2d(3, 32, 1),  # For 64x64 resolution
            SConv2d(3, 64, 1),  # For 32x32 resolution
            SConv2d(3, 128, 1),  # For 16x16 resolution
            SConv2d(3, 256, 1),  # For 8x8 resolution
            SConv2d(3, 512, 1)  # For 4x4 resolution
        ])
        
        # Assuming the highest resolution is 128x128, we define the downscale_blocks
        self.downscale_blocks = nn.ModuleList([
            DiscriminatorBlock(16, 32, 3, downsample=True),  # For 128x128 to 64x64
            DiscriminatorBlock(32, 64, 3, downsample=True),  # For 64x64 to 32x32
            DiscriminatorBlock(64, 128, 3, downsample=True),  # For 32x32 to 16x16
            DiscriminatorBlock(128, 256, 3, downsample=True),  # For 16x16 to 8x8
            DiscriminatorBlock(256, 512, 3, downsample=True),  # For 8x8 to 4x4
            DiscriminatorBlock(512, 512, 3, downsample=False)  # For 4x4 resolution
        ])
        
        self.len_layers = len(self.from_rgb_layers)
        
        self.final_block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling to 1x1 resolution
        )

        self.fc = nn.Linear(512, 1)  # Output a single scalar value
        
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
        
        x = self.final_block(x)
        # Flatten for fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        
        