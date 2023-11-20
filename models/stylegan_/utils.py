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
    batch_size = real_samples.shape[0]

    # generate random epsilon
    epsilon = torch.rand((batch_size, 1, 1, 1)).to(device)

    # create the merge of both real and fake samples
    merged = epsilon * real_samples + ((1 - epsilon) * fake_samples)
    merged.requires_grad_(True)

    # forward pass
    op = D(merged, level, alpha)

    # perform backward pass from op to merged for obtaining the gradients
    gradient = torch.autograd.grad(
        outputs=op,
        inputs=merged,
        grad_outputs=torch.ones_like(op),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)

    return ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()
