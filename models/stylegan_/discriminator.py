# File: discriminator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniBatchStdDev(nn.Module):
    """
    Mini-batch standard deviation layer for the discriminator.
    It helps the discriminator to use the variance of the batch to make decisions.
    """
    def forward(self, x):
        batch_size, _, height, width = x.shape
        std_dev = torch.std(x, dim=0).mean().repeat(batch_size, 1, height, width)
        return torch.cat([x, std_dev], dim=1)

class DiscriminatorBlock(nn.Module):
    """
    A block of the discriminator network.
    ...
    """
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(DiscriminatorBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, stride=stride, padding=padding)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        # Downsample the feature map conditionally
        if x.size(2) > 4 and x.size(3) > 4:
            x = F.avg_pool2d(x, 2)
        return x

class Discriminator(nn.Module):
    """
    StyleGAN Discriminator Network.
    ...
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        # Adjust the number of channels or layers based on the expected resolutions
        self.init_block = DiscriminatorBlock(3, 16)  # Initial block to handle the smallest resolution (4x4)
        self.blocks = nn.Sequential(
            DiscriminatorBlock(16, 32),
            DiscriminatorBlock(32, 64),
            DiscriminatorBlock(64, 128),
            DiscriminatorBlock(128, 256)
            # Additional blocks can be added for higher resolutions
        )
        self.mini_batch_std_dev = MiniBatchStdDev()
        self.final_conv = nn.Conv2d(257, 512, 3, padding=1)  # Adjust the input channel to 257 (256 + 1 for std dev)
        self.final_linear = nn.Linear(512, 1)

    def forward(self, x):
        print("Input shape: ", x.shape)
        x = self.init_block(x)
        print("After init block: ", x.shape)
        x = self.blocks(x)
        print("After blocks: ", x.shape)
        x = self.mini_batch_std_dev(x)
        print("After mini batch std dev: ", x.shape)
        x = self.final_conv(x)
        print("After final conv: ", x.shape)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        print("After adaptive avg pool: ", x.shape)
        x = x.view(x.size(0), -1)
        print("After view: ", x.shape)
        x = self.final_linear(x)
        print("After final linear: ", x.shape)
        print()
        return x
