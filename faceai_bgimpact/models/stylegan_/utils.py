import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

sqrt_2 = np.sqrt(2)


class WSConv2d(nn.Module):
    """Weight scaled convolutional layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, gain=sqrt_2):
        super().__init__()
        weight = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        init.normal_(weight)
        self.weight = nn.Parameter(weight)
        scale = gain / np.sqrt(in_channels * kernel_size * kernel_size)
        self.register_buffer("scale", torch.tensor(scale))

        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """Forward pass."""
        scaled_weight = self.weight * self.scale
        return F.conv2d(x, scaled_weight, self.bias, self.stride, self.padding)


class WSConvTranspose2d(nn.Module):
    """Weight scaled transposed convolutional layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, gain=sqrt_2):
        super().__init__()
        weight = torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        init.normal_(weight)
        self.weight = nn.Parameter(weight)
        scale = gain / np.sqrt(in_channels * kernel_size * kernel_size)
        self.register_buffer("scale", torch.tensor(scale))

        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """Forward pass."""
        scaled_weight = self.weight * self.scale
        return F.conv_transpose2d(x, scaled_weight, self.bias, self.stride, self.padding)


class PixelNorm(nn.Module):
    """Pixelwise feature vector normalization."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Forward pass."""
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class BlurLayer(nn.Module):
    """Blur operation for convolutional layers."""

    def __init__(self):
        super().__init__()

        f = np.array([1, 2, 1], dtype=np.float32)
        f = f[None, :] * f[:, None]
        f /= np.sum(f)
        f = f.reshape([1, 1, 3, 3])
        self.register_buffer("filter", torch.from_numpy(f))

    def forward(self, x):
        """Forward pass."""
        ch = x.size(1)
        return F.conv2d(x, self.filter.expand(ch, -1, -1, -1), padding=1, groups=ch)


class AdaIN(nn.Module):
    """Adaptive instance normalization."""

    def __init__(self, dim, w_dim):
        super().__init__()
        self.dim = dim
        self.epsilon = 1e-8
        self.scale_transform = WSConv2d(w_dim, dim, 1, 1, 0, gain=1)
        self.bias_transform = WSConv2d(w_dim, dim, 1, 1, 0, gain=1)

    def forward(self, x, w):
        """Forward pass."""
        x = F.instance_norm(x, eps=self.epsilon)

        # scale
        scale = self.scale_transform(w)
        bias = self.bias_transform(w)

        return scale * x + bias


class NoiseLayer(nn.Module):
    """Noise layer."""

    def __init__(self, n_channel, size):
        super().__init__()

        self.size = size
        self.register_buffer("fixed_noise", torch.randn([1, 1, size, size]))

        self.noise_scale = nn.Parameter(torch.zeros(1, n_channel, 1, 1))

    def forward(self, batch_size, device):
        """Forward pass."""
        noise = torch.randn([batch_size, 1, self.size, self.size], device=device)
        return noise * self.noise_scale


class MinibatchStdDev(nn.Module):
    """Minibatch standard deviation layer."""

    def __init__(self):
        super().__init__()

    def forward(self, x, group_size=4):
        """Forward pass."""
        batch_size, _, height, width = x.size()
        group_size = min(group_size, batch_size)  # Ensure group size is less than or equal to batch size
        if batch_size % group_size != 0:
            group_size = batch_size  # If batch size is not divisible by group_size, use the batch size
        stddev = x.view(group_size, -1, x.shape[1], x.shape[2], x.shape[3])
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([1, 2, 3], keepdim=True).squeeze(0)
        stddev = stddev.repeat(group_size, 1, height, width)
        return torch.cat([x, stddev], 1)
