import torch
from torch import nn

class PixelNorm(nn.Module):
    """Pixelwise feature vector normalization."""
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
    
def AdaIN(x, style):
    """Adaptive Instance Normalization."""
    mean, std = style.chunk(2, 1)
    std = std.exp()
    x = x.sub(x.mean([2, 3], keepdim=True)).div(x.std([2, 3], keepdim=True))
    x = std * x + mean
    return x

class NoiseInjection(nn.Module):
    """
    Noise injection layer.
    
    Parameters:
    ----------
    channel : int
        Number of input channels.
    """
    def __init__(self, channel):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        # Print the shapes of everything
        print(f"Image shape: {image.shape}")
        print(f"Noise shape: {noise.shape}")
        print(f"Weight shape: {self.weight.shape}")
        return image + self.weight * noise