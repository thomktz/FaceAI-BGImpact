import torch
import pytest
from models.stylegan import Generator, SynthesisNetwork, MappingNetwork

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
            #(0, 1.0), 
            (1, 0.5), 
            #(2, 1.0), 
            (3, 0.5), 
            #(4, 1.0)
        ]
    )
    def test_forward_pass(self, level, alpha):
        """Test the forward pass of StyleGAN at different levels."""
        generator = Generator(self.latent_dim, self.w_dim, self.style_layers, self.image_size)
        z = torch.randn(1, self.latent_dim)  # Single latent vector

        with torch.no_grad():
            image = generator(z, level, alpha)

        expected_size = 4 * 2 ** level  # The expected size of the image at the current level
        assert image.shape == (1, 3, expected_size, expected_size)
