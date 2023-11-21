import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, gain=np.sqrt(2)):
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
        scaled_weight = self.weight * self.scale
        return F.conv2d(x, scaled_weight, self.bias, self.stride, self.padding)


class WSLinear(nn.Module):
    def __init__(
        self, in_features, out_features, gain=2,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (gain / in_features)**0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        # initialize linear layer
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias
    
class WSConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, gain=np.sqrt(2)):
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
        scaled_weight = self.weight * self.scale
        return F.conv_transpose2d(x, scaled_weight, self.bias, self.stride, self.padding)

class PixelNorm(nn.Module):
    """Pixelwise feature vector normalization."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
class BlurLayer(nn.Module):
    def __init__(self):
        super().__init__()

        f = np.array([1, 2, 1], dtype=np.float32)
        f = f[None, :] * f[:, None]
        f /= np.sum(f)
        f = f.reshape([1, 1, 3, 3])
        self.register_buffer("filter", torch.from_numpy(f))

    def forward(self, x):
        ch = x.size(1)
        return F.conv2d(x, self.filter.expand(ch, -1, -1, -1), padding=1, groups=ch)

class FC_A(nn.Module):
    '''
    Learned affine transform "A" to transform the latent vector w into a style vector y
    '''
    def __init__(self, dim_latent, n_channel):
        super().__init__()
        self.transform = WSLinear(dim_latent, n_channel * 2)
        # "the biases associated with ys that we initialize to one"
        self.transform.bias.data[:n_channel] = 1
        self.transform.bias.data[n_channel:] = 0

    def forward(self, w):
        # Gain scale factor and bias with:
        style = self.transform(w).unsqueeze(2).unsqueeze(3)
        return style
    
class AdaIN(nn.Module):
    def __init__(self, dim, w_dim):
        super().__init__()
        self.dim = dim
        self.epsilon = 1e-8
        self.scale_transform = WSConv2d(w_dim, dim, 1, 1, 0, gain=1)
        self.bias_transform = WSConv2d(w_dim, dim, 1, 1, 0, gain=1)

    def forward(self, x, w):
        x = F.instance_norm(x, eps=self.epsilon)

        # scale
        scale = self.scale_transform(w)
        bias = self.bias_transform(w)

        return scale * x + bias

class NoiseLayer(nn.Module):
    def __init__(self, n_channel, size):
        super().__init__()

        self.fixed = False

        self.size = size
        self.register_buffer("fixed_noise", torch.randn([1, 1, size, size]))

        self.noise_scale = nn.Parameter(torch.zeros(1, n_channel, 1, 1))

    def forward(self, batch_size, device):
        if self.fixed:
            noise = self.fixed_noise.expand(batch_size, -1, -1, -1)
        else:
            noise = torch.randn([batch_size, 1, self.size, self.size], device=device)
        return noise * self.noise_scale
