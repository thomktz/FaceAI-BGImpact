import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, latent_dim) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )

        self.fc_mu = nn.Linear(128 * 32 * 32, latent_dim)
        self.fc_logvar = nn.Linear(128 * 32 * 32, latent_dim)
        self.relu = nn.ReLU
    
    def forward(self, z):
        """
        Forward pass for the generator.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (batch_size, 3, 128, 128).

        Returns
        -------
        mu : torch.Tensor
            Mean of the latent space
        logvar : torch.Tensor 
            Log-variance of the latent space
        """
        out = self.conv_blocks(z)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar


    