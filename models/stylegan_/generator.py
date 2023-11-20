import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import pairwise_euclidean_distance
from .utils import PixelNorm, AdaIN, FC_A, Scale_B, SConv2d, scale_module

class MappingNetwork(nn.Module):
    """
    Mapping Network.
    
    Parameters:
    ----------
    latent_dim : int
        Dimension of the latent vectors.
    layers : int
        Number of layers in the mapping network.
    w_dim : int
        Dimension of the intermediate noise vector in W space.    
    """
    
    def __init__(self, latent_dim, layers, w_dim):
        super(MappingNetwork, self).__init__()
        
        model = [nn.Linear(latent_dim, w_dim), PixelNorm(), nn.LeakyReLU(0.2)]
        for _ in range(layers - 1):
            model.extend([nn.Linear(w_dim, w_dim), PixelNorm(), nn.LeakyReLU(0.2)])
        self.model = nn.Sequential(*model)

    def forward(self, z):
        """
        Forward pass for the mapping network.
        
        Parameters:
        ----------
        z (torch.Tensor): Input tensor.
            Shape: (batch_size, latent_dim)
        
        Returns:
        ----------
        torch.Tensor: Output tensor.
            Shape: (batch_size, w_dim)
        """
        w = self.model(z)
        return w

class StyledConvBlock(nn.Module):
    """
    StyleGAN convolutional block.
    
    Parameters:
    ----------
    in_channel : int
        Number of input channels.
    out_channel : int
        Number of output channels.
    style_dim : int
        Dimension of the style vector.
    """
    def __init__(self, in_channel, out_channel, style_dim, is_first_block=False):
        super().__init__()
        self.is_first_block = is_first_block
        
        self.style1 = FC_A(style_dim, out_channel)
        self.style2 = FC_A(style_dim, out_channel)
        
        self.noise1 = scale_module(Scale_B(out_channel))
        self.noise2 = scale_module(Scale_B(out_channel))
        
        self.adain = AdaIN(out_channel)
        self.act = nn.LeakyReLU(0.2)
        
        if not is_first_block:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv1 = SConv2d(in_channel, out_channel, 3, padding=1)
            self.conv2 = SConv2d(out_channel, out_channel, 3, padding=1)
        else:
            self.conv = SConv2d(in_channel, out_channel, 3, padding=1)

    def forward(self, x, w, noise):
        """
        Forward pass for the StyleGAN convolutional block.
        
        Parameters and dimensions:
        ----------
        x (torch.Tensor): Input tensor.
            Shape: (batch_size, in_channel, height, width)
        w (torch.Tensor): Style tensor.
            Shape: (batch_size, style_dim)
        noise (torch.Tensor): Noise tensor.
            Shape: (batch_size, 1, height, width)
            
        Returns:
        ----------
        torch.Tensor: Output tensor.
            Shape: (batch_size, out_channel, height, width)
        """
        if not self.is_first_block:
            x = self.upsample(x)
            x = self.conv1(x)

        x = x + self.noise1(noise)
        x = self.adain(x, self.style1(w))
        x = self.act(x)
        
        if not self.is_first_block:
            x = self.conv2(x)
        else:
            x = self.conv(x)
            
        x = x + self.noise2(noise)
        x = self.adain(x, self.style2(w))
        x = self.act(x)

        return x

class SynthesisNetwork(nn.Module):
    """
    Generator Synthesis Network.
    
    Parameters:
    ----------
    w_dim : int
        Dimension of the intermediate noise vector in W space.
    """
    
    def __init__(self, w_dim):
        super(SynthesisNetwork, self).__init__()
        self.init_size = 4  # Initial resolution
        self.learned_constant = nn.Parameter(torch.randn(1, 512, 4, 4)) # 'x' for the init_block, learned constant
        self.init_block = StyledConvBlock(w_dim, 512, w_dim, is_first_block=True)  # Initial block

        # Sequentially larger blocks for higher resolutions
        self.upscale_blocks = nn.ModuleList([
            StyledConvBlock(512, 256, w_dim, is_first_block=False),  # 8x8
            StyledConvBlock(256, 128, w_dim, is_first_block=False),  # 16x16
            StyledConvBlock(128, 64, w_dim, is_first_block=False),   # 32x32
            StyledConvBlock(64, 32, w_dim, is_first_block=False),    # 64x64
            StyledConvBlock(32, 16, w_dim, is_first_block=False)     # 128x128
        ])

        # To-RGB layers for each resolution
        self.to_rgb_layers = nn.ModuleList([
            SConv2d(512, 3, 1, stride=1, padding=0),
            SConv2d(256, 3, 1, stride=1, padding=0),
            SConv2d(128, 3, 1, stride=1, padding=0),
            SConv2d(64, 3, 1, stride=1, padding=0),
            SConv2d(32, 3, 1, stride=1, padding=0),
            SConv2d(16, 3, 1, stride=1, padding=0)
        ])

    def forward(self, w, current_level, alpha):
        """
        Forward pass with progressive growing.

        Parameters:
        ----------
        w : torch.Tensor
            Style tensor.
            Shape: (batch_size, w_dim)
        current_level : int
            Current resolution level for progressive growing.
        alpha : float
            Blending factor for progressive growing.
        
        Returns:
        ----------
        torch.Tensor: Generated image tensor.
        """

        x = self.learned_constant.repeat(w.shape[0], 1, 1, 1)
        noise = torch.randn(w.shape[0], 1, self.init_size, self.init_size, device=w.device)
        
        x = self.init_block(x, w, noise)
        print("Pairwise euclidian distance in the x batch:", pairwise_euclidean_distance(x).item())
        # Get the initial RGB image at 4x4 resolution
        if current_level <= 1:
            rgb = self.to_rgb_layers[0](x)

        for level in range(1, current_level + 1):
            level_resolution = 4 * 2 ** level
            noise = torch.randn(w.shape[0], 1, level_resolution, level_resolution, device=w.device)
            
            x = self.upscale_blocks[level - 1](x, w, noise)
        
            if alpha < 1.0 and level == current_level:
                # Interpolate between the new RGB image of the current resolution
                # and the upscaled RGB image of the previous resolution
                new_rgb = self.to_rgb_layers[level](x)
                rgb = F.interpolate(rgb, scale_factor=2, mode='nearest')
                rgb = alpha * new_rgb + (1 - alpha) * rgb 
            else:
                rgb = self.to_rgb_layers[level](x)
        return rgb


class Generator(nn.Module):
    """
    StyleGAN Generator Network.
    """
    def __init__(self, latent_dim, w_dim, style_layers):
        super(Generator, self).__init__()
        self.mapping = MappingNetwork(latent_dim, style_layers, w_dim)
        self.synthesis = SynthesisNetwork(w_dim)

    def forward(self, z, current_level, alpha):
        """
        Forward pass for the StyleGAN generator.

        Parameters:
        ----------
        z : torch.Tensor
            A batch of latent vectors.
        current_level : int
            Current resolution level for progressive growing.
        alpha : float
            Blending factor for progressive growing.

        Returns:
        ----------
        torch.Tensor: Generated image tensor.
        """
        
        print("Pairwise euclidian distance in the z batch:", pairwise_euclidean_distance(z).item())
        w = self.mapping(z)
        image = self.synthesis(w, current_level, alpha)
        return image
