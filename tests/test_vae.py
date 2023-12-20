# flake8: noqa

import torch
import pytest
from faceai_bgimpact.models.vae_ import VAE


class TestVAE:
    latent_dim = 128

    @pytest.fixture
    def sample_vae(self):
        return VAE(dataset_name="ffhq_raw", latent_dim=self.latent_dim, device="cuda" if torch.cuda.is_available() else "cpu")

    def test_initialization(self, sample_vae):
        """Test the initialization of VAE components."""
        assert sample_vae.encoder is not None, "Encoder not initialized"
        assert sample_vae.decoder is not None, "Decoder not initialized"
    
    def test_reparameterize(self, sample_vae):
        """Test the reparameterization of the VAE model"""
        mu = torch.zeros(1,self.latent_dim)
        logvar = torch.zeros(1,self.latent_dim)
        z = sample_vae.reparameterize(mu, logvar)
        
        assert z.shape == (1, self.latent_dim), "Reparametrization dimensions are incorrect"

    def test_generated_image_dimensions(self, sample_vae):
        """Test the dimension of a generated image from a VAE model."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        z = torch.rand(1, 3, 128, 128).to(device)
        mu, logvar = sample_vae.encoder(z)
        z_sample = sample_vae.reparameterize(mu, logvar)
        fake_image = sample_vae.decoder(z_sample).cpu()
        
        assert fake_image.shape == (1,3,128,128), "Generated image dimensions are incorrect"

    def test_forward_pass_encoder_vae(self, sample_vae):
        """Test the forward pass of VAE Encoder."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        z = torch.rand(1, 3, 128, 128).to(device)
        mu, logvar = sample_vae.encoder(z)
        
        assert mu.shape == (1, self.latent_dim), "Mu encoded dimension after forward pass is incorrect"
        assert logvar.shape == (1,self.latent_dim), "Variance encoded dimension after forward pass is incorrect"

    def test_forward_pass_decoder_vae(self, sample_vae):
        """Test the forward pass of VAE Decoder."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        z = torch.rand(1, self.latent_dim).to(device)
        fake_image = sample_vae.decoder(z)
        
        assert fake_image.shape == (1,3,128,128), "Generated image dimensions after forward pass are incorrect"