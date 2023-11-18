import torch
import pytest
from models.stylegan import Generator, SynthesisNetwork, MappingNetwork, Discriminator

class TestStyleGAN:
    latent_dim = 100
    w_dim = 512
    style_layers = 8
    image_size = 128  # Assuming target image size is 128x128

    def test_initialization(self):
        """Test the initialization of StyleGAN components."""
        mapping_network = MappingNetwork(self.latent_dim, self.style_layers, self.w_dim)
        synthesis_network = SynthesisNetwork(self.w_dim)
        generator = Generator(self.latent_dim, self.w_dim, self.style_layers, self.image_size)

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
        generator = Generator(self.latent_dim, self.w_dim, self.style_layers, self.image_size)
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
        "resolution", 
        [
            (4, 4), 
            (8, 8), 
            (16, 16), 
            (32, 32), 
            (64, 64),
            (128, 128)
        ]
    )
    def test_forward_pass(self, resolution):
        """Test the forward pass of StyleGAN Discriminator at different resolutions."""
        discriminator = Discriminator()
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, *resolution)  # Random image tensor

        with torch.no_grad():
            output = discriminator(input_tensor)

        assert output.shape == (batch_size, 1)  # Expecting a single value per image in the batch
