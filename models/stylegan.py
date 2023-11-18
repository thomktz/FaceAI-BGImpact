import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchvision.utils import save_image
from pytorch_gan_metrics import get_fid

from models.abstract_model import AbstractModel
from models.data_loader import get_dataloader, denormalize_imagenet
from models.utils import weights_init

from models.stylegan_.generator import Generator
from models.stylegan_.discriminator import Discriminator


class StyleGAN(AbstractModel):
    def __init__(self, latent_dim, w_dim, style_layers):
        super(StyleGAN, self).__init__()
        
        self.latent_dim = latent_dim
        self.w_dim = w_dim
        self.style_layers = style_layers
        
        self.generator = Generator(self.latent_dim, self.w_dim, self.style_layers)
        self.discriminator = Discriminator()

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
