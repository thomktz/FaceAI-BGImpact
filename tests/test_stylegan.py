import torch
import pytest
from models.stylegan_ import Generator, SynthesisNetwork, MappingNetwork, Discriminator

class TestStyleGAN:
    latent_dim = 100
    w_dim = 512
    style_layers = 8

    def test_initialization(self):
        """Test the initialization of StyleGAN components."""
        mapping_network = MappingNetwork(self.latent_dim, self.style_layers, self.w_dim)
        synthesis_network = SynthesisNetwork(self.w_dim)
        generator = Generator(self.latent_dim, self.w_dim, self.style_layers)

        assert mapping_network is not None
        assert synthesis_network is not None
        assert generator is not None

    @pytest.mark.parametrize(
        "level,alpha", 
        [
            (0, 1.0), 
            (1, 0.5), 
            (2, 1.0), 
            (3, 0.5), 
            (4, 1.0),
            (5, 0.5)
        ]
    )
    def test_forward_pass(self, level, alpha):
        """Test the forward pass of StyleGAN at different levels."""
        generator = Generator(self.latent_dim, self.w_dim, self.style_layers)
        batch_size = 2
        z = torch.randn(batch_size, self.latent_dim)  # Single latent vector

        with torch.no_grad():
            image = generator(z, level, alpha)

        expected_size = 4 * 2 ** level  # The expected size of the image at the current level
        assert image.shape == (batch_size, 3, expected_size, expected_size)


class TestStyleGANDiscriminator:
    def test_initialization(self):
        """Test the initialization of StyleGAN Discriminator."""
        discriminator = Discriminator()
        assert discriminator is not None

    @pytest.mark.parametrize(
        "alpha,current_level", 
        [
            (1.0, 0), 
            (0.5, 1), 
            (1.0, 1), 
            (0.5, 2), 
            (1.0, 2),
            (0.5, 3),
            (1.0, 3),
            (0.5, 4),
            (1.0, 4),
            (0.5, 5),
            (1.0, 5)
        ]
    )
    def test_forward_pass(self, alpha, current_level):
        """Test the forward pass of StyleGAN Discriminator at different resolutions."""
        discriminator = Discriminator()
        batch_size = 2
        resolution = 4 * 2 ** current_level
        image = torch.randn(batch_size, 3, resolution, resolution)

        with torch.no_grad():
            output = discriminator(image, current_level, alpha)

        assert output.shape == (batch_size, 1)
