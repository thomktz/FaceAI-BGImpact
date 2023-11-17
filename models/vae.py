""" SOURCES : https://arxiv.org/pdf/1312.6114.pdf
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchvision.utils import save_image
# from pytorch_gan_metrics import get_fid

from models.abstract_model import AbstractModel
from models.data_loader import get_dataloader, denormalize_imagenet


class Encoder(nn.Module):
    """
    Encoder class for the VAE.
    
    Parameters
    ----------
    latent_dim : int
        Dimension of the latent space. Defaults to 100.
    """

    def __init__(self, latent_dim):
        super(Encoder, self).__init__()



    def forward(self, z):
        """
        Forward pass for the encoder.
        
        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (batch_size, latent_dim).
            
        Returns
        -------
        img : torch.Tensor
            Output tensor of shape (batch_size, 3, 128, 128).
        """
        
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img