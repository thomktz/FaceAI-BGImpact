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
        return image + self.weight * noise
    
def compute_gradient_penalty(D, real_samples, fake_samples, level, alpha, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    # Not the same alpha as the interpolation factor for the resolution level blending
    alpha_ = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha_ * real_samples + ((1 - alpha_) * fake_samples)).requires_grad_(True)
    
    d_interpolates = D(interpolates, level, alpha)
    fake = torch.ones(real_samples.shape[0], 1, requires_grad=False, device=device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    # Add epsilon as per https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
    gradient_penalty = ((gradients.norm(2, dim=1) - 1 + 1e-12) ** 2).mean()
    return gradient_penalty
