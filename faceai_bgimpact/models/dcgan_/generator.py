import torch.nn as nn

class Generator(nn.Module):
    """
    Generator class for the DCGAN.
    
    Parameters
    ----------
    latent_dim : int
        Dimension of the latent space. Defaults to 100.
    """
    
    def __init__(self, latent_dim):
        super().__init__()
        self.init_size = 128 // 4  # Initial size before upsampling
        
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        """
        Forward pass for the generator.
        
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