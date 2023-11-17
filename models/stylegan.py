import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import PixelNorm, AdaIN, NoiseInjection


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
        model = [nn.Linear(latent_dim, w_dim), nn.LeakyReLU(0.2)]
        for _ in range(layers - 1):
            model.append(nn.Linear(w_dim, w_dim))
            model.append(nn.LeakyReLU(0.2))
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
    kernel_size : int
        Kernel size.
    style_dim : int
        Dimension of the style vector.
    """
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False):
        super(StyledConvBlock, self).__init__()
        # if upsample:
        #     self.conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride=1, padding=kernel_size//2, output_padding=1)
        # else:
        #     self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=1, padding=kernel_size//2)
        if upsample:
            self.conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride=2, padding=1, output_padding=1)
        else:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=1, padding=1)
        self.style = nn.Linear(style_dim, out_channel * 2)
        self.noise = NoiseInjection(out_channel)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, style):
        """
        Forward pass for the StyleGAN convolutional block.
        
        Parameters and dimensions:
        ----------
        x (torch.Tensor): Input tensor.
            Shape: (batch_size, in_channel, height, width)
        style (torch.Tensor): Style vector.
            Shape: (batch_size, style_dim)
            
        Returns:
        ----------
        torch.Tensor: Output tensor.
            Shape: (batch_size, out_channel, height, width)
        """
        print(f"StyledConvBlock - Input x shape: {x.shape}")
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        x = self.conv(x)  
        print(f"StyledConvBlock - After conv x shape: {x.shape}")

        batch, _, height, width = x.shape
        noise = torch.randn(batch, 1, height, width, device=x.device)
        x = self.noise(x, noise)
        print(f"StyledConvBlock - After noise x shape: {x.shape}")

        x = AdaIN(x, style)
        x = self.act(x)
        print(f"StyledConvBlock - Output x shape: {x.shape}")
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
        self.init_block = StyledConvBlock(w_dim, 512, 3, w_dim, upsample=False)  # Initial block

        # Sequentially larger blocks for higher resolutions
        self.upscale_blocks = nn.ModuleList([
            StyledConvBlock(512, 256, 3, w_dim, upsample=True),  # 8x8
            StyledConvBlock(256, 128, 3, w_dim, upsample=True),  # 16x16
            StyledConvBlock(128, 64, 3, w_dim, upsample=True),   # 32x32
            StyledConvBlock(64, 32, 3, w_dim, upsample=True),    # 64x64
            StyledConvBlock(32, 16, 3, w_dim, upsample=True)     # 128x128
        ])

        # To-RGB layers for each resolution
        self.to_rgb_layers = nn.ModuleList([
            nn.Conv2d(512, 3, 1, stride=1, padding=0),
            nn.Conv2d(256, 3, 1, stride=1, padding=0),
            nn.Conv2d(128, 3, 1, stride=1, padding=0),
            nn.Conv2d(64, 3, 1, stride=1, padding=0),
            nn.Conv2d(32, 3, 1, stride=1, padding=0),
            nn.Conv2d(16, 3, 1, stride=1, padding=0)
        ])

    def forward(self, w, current_level, alpha):
        """
        Forward pass with progressive growing.

        Parameters:
        w (torch.Tensor): Input tensor in W space.
        current_level (int): Current resolution level.
        alpha (float): Blending factor for progressive growing.
        
        Returns:
        torch.Tensor: Generated image tensor.
        """
        x = self.learned_constant.repeat(w.shape[0], 1, 1, 1)  
        print(f"SynthesisNetwork - Initial x shape: {x.shape}")
        x = self.init_block(x, w)

        if current_level == 0:
            output = self.to_rgb_layers[0](x)
            print(f"SynthesisNetwork - Output at level 0 shape: {output.shape}")
            return output

        for level in range(current_level):
            x = self.upscale_blocks[level](x, w)
            print(f"SynthesisNetwork - After level {level} x shape: {x.shape}")

        if alpha < 1.0 and current_level > 0:
            skip_rgb = self.to_rgb_layers[current_level - 1](x)
            skip_rgb = F.interpolate(skip_rgb, scale_factor=2, mode='nearest')
            x = self.upscale_blocks[current_level](x, w)
            direct_rgb = self.to_rgb_layers[current_level](x)
            output = (1 - alpha) * skip_rgb + alpha * direct_rgb
        else:
            output = self.to_rgb_layers[current_level](x)

        print(f"SynthesisNetwork - Final output shape: {output.shape}")
        return output


class Generator(nn.Module):
    """
    StyleGAN Generator Network.
    """
    def __init__(self, latent_dim, w_dim, style_layers, image_size):
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
        print(f"Generator - w shape: {w.shape}")
        image = self.synthesis(w, current_level, alpha)
        print(f"Generator - Final image shape: {image.shape}")
        return image
