import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder class for the VAE.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent space.
    """

    def __init__(self, latent_dim):
        super().__init__()
        self.init_size = 128 // 8

        self.fc = nn.Linear(latent_dim, 128 * self.init_size**2)

        self.deconv_blocks = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z):
        """
        Forward pass for the decoder.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (batch_size, latent_dim).

        Returns
        -------
        x_recon : torch.Tensor
            Reconstructed output tensor of shape (batch_size, 3, 128, 128).
        """
        out = self.fc(z)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        out_recon = self.deconv_blocks(out)
        return out_recon
