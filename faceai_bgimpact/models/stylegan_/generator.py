import torch
import torch.nn as nn
import torch.nn.functional as F
from faceai_bgimpact.models.utils import pairwise_euclidean_distance
from faceai_bgimpact.models.stylegan_.utils import PixelNorm, AdaIN, BlurLayer, NoiseLayer, WSConv2d

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
        model = [PixelNorm()]
        for _ in range(layers):
            model.extend([WSConv2d(latent_dim, latent_dim, 1, 1, 0), nn.LeakyReLU(0.2)])
        model.extend([WSConv2d(latent_dim, w_dim, 1, 1, 0)])
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
        # Expand the latent vector to a 4D tensor
        z = z.unsqueeze(2).unsqueeze(3)
        # Get the intermediate noise vector in W space
        w = self.model(z)
        return w

class SynthesisBlock(nn.Module):
    """
    StyleGAN convolutional block.
    
    Parameters:
    ----------
    in_channel : int
        Number of input channels.
    out_channel : int
        Number of output channels.
    size : int
        Resolution size of the input tensor.
    is_first_block : bool
        Whether this is the first block in the network.
    """
    def __init__(self, in_channel, out_channel, w_dim, size, is_first_block=False):
        super().__init__()
        self.is_first_block = is_first_block
        
        self.noise1 = NoiseLayer(out_channel, size)
        self.noise2 = NoiseLayer(out_channel, size)
        
        self.blur = BlurLayer()
        
        self.adain = AdaIN(out_channel, w_dim)
        self.act = nn.LeakyReLU(0.2)
        
        if not is_first_block:
            self.conv1 = WSConv2d(in_channel, out_channel, 3, 1, 1)
            self.conv2 = WSConv2d(out_channel, out_channel, 3, 1, 1)
        else:
            self.conv = WSConv2d(in_channel, out_channel, 3, 1, 1)

    def forward(self, x, w):
        """
        Forward pass for the StyleGAN convolutional block.
        
        Parameters and dimensions:
        ----------
        x (torch.Tensor): Input tensor.
            Shape: (batch_size, in_channel, height, width)
        w (torch.Tensor): Style tensor.
            Shape: (batch_size, style_dim)
            
        Returns:
        ----------
        torch.Tensor: Output tensor.
            Shape: (batch_size, out_channel, height, width)
        """
        #TODO: w1 and w2
        if not self.is_first_block:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', antialias=True)
            x = self.conv1(x)

        x = x + self.noise1(x.shape[0], x.device)
        x = self.adain(x, w)
        x = self.act(x)
        
        if not self.is_first_block:
            x = self.conv2(x)
        else:
            x = self.conv(x)
        
        x = x + self.noise2(x.shape[0], x.device)
        x = self.adain(x, w)
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
        self.learned_constant = nn.Parameter(torch.randn(1, 256, 4, 4)) # 'x' for the init_block, learned constant
        self.init_block = SynthesisBlock(w_dim, 256, w_dim, 4, is_first_block=True)  # Initial block

        # Sequentially larger blocks for higher resolutions
        self.upscale_blocks = nn.ModuleList([
            SynthesisBlock(256, 256, w_dim, 8, is_first_block=False),  # 8x8
            SynthesisBlock(256, 128, w_dim, 16, is_first_block=False), # 16x16
            SynthesisBlock(128, 64, w_dim, 32, is_first_block=False),  # 32x32
            SynthesisBlock(64, 32, w_dim, 64, is_first_block=False),   # 64x64
            SynthesisBlock(32, 16, w_dim, 128, is_first_block=False)   # 128x128
        ])

        # To-RGB layers for each resolution
        self.to_rgb_layers = nn.ModuleList([
            WSConv2d(256, 3, 1, 1, 0, gain=1),
            WSConv2d(256, 3, 1, 1, 0, gain=1),
            WSConv2d(128, 3, 1, 1, 0, gain=1),
            WSConv2d(64, 3, 1, 1, 0, gain=1),
            WSConv2d(32, 3, 1, 1, 0, gain=1),
            WSConv2d(16, 3, 1, 1, 0, gain=1)
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
        
        x = self.init_block(x, w)
        
        # Get the initial RGB image at 4x4 resolution
        if current_level <= 1:
            rgb = self.to_rgb_layers[0](x)

        for level in range(1, current_level + 1):
            
            x = self.upscale_blocks[level - 1](x, w)
        
            if alpha < 1.0 and level == current_level:
                # Interpolate between the new RGB image of the current resolution
                # and the upscaled RGB image of the previous resolution
                new_rgb = self.to_rgb_layers[level](x)
                rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', antialias=True)
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
    
        w = self.mapping(z)
        image = self.synthesis(w, current_level, alpha)
        return image
