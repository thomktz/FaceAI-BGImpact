# test_gan.py
import pytest
import torch
from models.gan import GAN
from models.data_loader import get_dataloaders

@pytest.fixture
def sample_gan():
    return GAN(dataset="ffhq_raw", lr=0.0002, latent_dim=100, batch_size=10)

def test_gan_initialization(sample_gan):
    assert sample_gan.generator is not None, "Generator not initialized"
    assert sample_gan.discriminator is not None, "Discriminator not initialized"

def test_gan_train_step(sample_gan):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 1
    test_batches_limit = 50  # Limit the number of batches for testing


    sample_gan.train(num_epochs=num_epochs, device=device, test_batches_limit=test_batches_limit)

    # Check if grads are updated
    assert sample_gan.generator_optimizer.param_groups[0]['params'][0].grad is not None, "Generator grads not updated"
    assert sample_gan.discriminator_optimizer.param_groups[0]['params'][0].grad is not None, "Discriminator grads not updated"

def test_gan_generated_image_dimensions(sample_gan):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = torch.randn(1, sample_gan.latent_dim).to(device)
    fake_image = sample_gan.generator(z)
    assert fake_image.shape == (1, 3, 128, 128), "Generated image dimensions are incorrect"
