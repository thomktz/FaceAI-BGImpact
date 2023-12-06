import torch
import torch.nn as nn
import torch.nn.functional as F
from faceai_bgimpact.models.stylegan_.utils import (
    PixelNorm,
    AdaIN,
    BlurLayer,
    NoiseLayer,
    WSConv2d,
)


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
        Dimension of the intermediate style vector in W space.
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
        # Get the intermediate style vector in W space
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

    def forward(self, x, w1, w2, apply_noise):
        """
        Forward pass for the StyleGAN convolutional block.

        Parameters and dimensions:
        ----------
        x (torch.Tensor): Input tensor.
            Shape: (batch_size, in_channel, height, width)
        w1 (torch.Tensor): Style tensor.
            Shape: (batch_size, style_dim)
        w2 (torch.Tensor): Style tensor. Will be the same as w1 except in layer analysis.
            Shape: (batch_size, style_dim)
        apply_noise (bool): Whether to add noise to the input tensor.

        Returns:
        ----------
        torch.Tensor: Output tensor.
            Shape: (batch_size, out_channel, height, width)
        """
        if w2 is None:
            w2 = w1
        if not self.is_first_block:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", antialias=True)
            x = self.conv1(x)

        if apply_noise:
            x = x + self.noise1(x.shape[0], x.device)
        x = self.adain(x, w1)
        x = self.act(x)

        if not self.is_first_block:
            x = self.conv2(x)
        else:
            x = self.conv(x)
        if apply_noise:
            x = x + self.noise2(x.shape[0], x.device)
        x = self.adain(x, w2)
        x = self.act(x)

        return x


class SynthesisNetwork(nn.Module):
    """
    Generator Synthesis Network.

    Parameters:
    ----------
    w_dim : int
        Dimension of the intermediate style vector in W space.
    """

    def __init__(self, w_dim):
        super(SynthesisNetwork, self).__init__()
        self.init_size = 4  # Initial resolution
        self.learned_constant = nn.Parameter(torch.randn(1, 256, 4, 4))  # 'x' for the init_block, learned constant
        self.init_block = SynthesisBlock(w_dim, 256, w_dim, 4, is_first_block=True)  # Initial block

        # Sequentially larger blocks for higher resolutions
        self.upscale_blocks = nn.ModuleList(
            [
                SynthesisBlock(256, 256, w_dim, 8, is_first_block=False),  # 8x8
                SynthesisBlock(256, 128, w_dim, 16, is_first_block=False),  # 16x16
                SynthesisBlock(128, 64, w_dim, 32, is_first_block=False),  # 32x32
                SynthesisBlock(64, 32, w_dim, 64, is_first_block=False),  # 64x64
                SynthesisBlock(32, 16, w_dim, 128, is_first_block=False),  # 128x128
            ]
        )

        # To-RGB layers for each resolution
        self.to_rgb_layers = nn.ModuleList(
            [
                WSConv2d(256, 3, 1, 1, 0, gain=1),
                WSConv2d(256, 3, 1, 1, 0, gain=1),
                WSConv2d(128, 3, 1, 1, 0, gain=1),
                WSConv2d(64, 3, 1, 1, 0, gain=1),
                WSConv2d(32, 3, 1, 1, 0, gain=1),
                WSConv2d(16, 3, 1, 1, 0, gain=1),
            ]
        )

    def forward(self, w, current_level, alpha, apply_noise):
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
        apply_noise : bool
            Whether to add noise to the input tensor.

        Returns:
        ----------
        torch.Tensor: Generated image tensor.
        """
        x = self.learned_constant.repeat(w.shape[0], 1, 1, 1)

        x = self.init_block(x, w, w, apply_noise=apply_noise)

        # Get the initial RGB image at 4x4 resolution
        if current_level <= 1:
            rgb = self.to_rgb_layers[0](x)

        for level in range(1, current_level + 1):
            x = self.upscale_blocks[level - 1](x, w, w, apply_noise=apply_noise)

            if alpha < 1.0 and level == current_level:
                # Interpolate between the new RGB image of the current resolution
                # and the upscaled RGB image of the previous resolution
                new_rgb = self.to_rgb_layers[level](x)
                rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear", antialias=True)
                rgb = alpha * new_rgb + (1 - alpha) * rgb
            else:
                rgb = self.to_rgb_layers[level](x)
        return rgb

    def predict_modified_layer(self, new_ws, current_level, apply_noise):
        """
        Generate an image using base_w for all layers except specified layers where new_w is used.

        Assumes alpha = 1.0.

        Parameters:
        ----------
        new_ws : torch.Tensor
            New style tensor to be used in specified layers.
            Shape: (n_layers, batch_size, w_dim)
        new_w_layers : list of int
            List of layer indices where new_w should be used.
        current_level : int
            Current resolution level for progressive growing.
        apply_noise : bool
            Whether to add noise to the input tensor.

        Returns:
        ----------
        torch.Tensor: Generated image tensor.
        """
        w1 = new_ws[0].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        w2 = new_ws[1].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = self.learned_constant.repeat(w1.shape[0], 1, 1, 1)

        x = self.init_block(x, w1, w2, apply_noise=apply_noise)

        for level in range(1, current_level + 1):
            w1 = new_ws[2 * level].unsqueeze(0).unsqueeze(2).unsqueeze(3)
            w2 = new_ws[2 * level + 1].unsqueeze(0).unsqueeze(2).unsqueeze(3)
            x = self.upscale_blocks[level - 1](x, w1, w2, apply_noise=apply_noise)

            if level == current_level:
                rgb = self.to_rgb_layers[level](x)

        return rgb


class Generator(nn.Module):
    """StyleGAN Generator Network."""

    def __init__(self, latent_dim, w_dim, style_layers):
        super(Generator, self).__init__()
        self.mapping = MappingNetwork(latent_dim, style_layers, w_dim)
        self.synthesis = SynthesisNetwork(w_dim)

    def forward(self, z, current_level, alpha, apply_noise=True):
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
        apply_noise : bool
            Whether to add noise to the input tensor.

        Returns:
        ----------
        torch.Tensor: Generated image tensor.
        """
        w = self.mapping(z)
        image = self.synthesis(w, current_level, alpha, apply_noise)
        return image

    def predict_from_style(self, w, current_level, alpha, apply_noise):
        """
        Generate images from a given style vector 'w' and optional noise.

        Parameters:
        ----------
        w : torch.Tensor
            A batch of style vectors.
        current_level : int
            Current resolution level for progressive growing.
        alpha : float
            Blending factor for progressive growing.
        apply_noise : bool
            Whether to add noise to the input tensor.

        Returns:
        ----------
        torch.Tensor: Generated image tensor.
        """
        image = self.synthesis(w, current_level, alpha, apply_noise)
        return image
