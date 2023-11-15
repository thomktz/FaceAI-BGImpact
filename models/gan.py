import os
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from tqdm import tqdm
from torchvision.utils import save_image

from models.data_loader import get_dataloaders
from models.abstract_model import AbstractModel

DEFAULT_LATENT_DIM = 100

class Generator(nn.Module):
    """
    Generator class for the GAN.
    
    Parameters
    ----------
    latent_dim : int
        Dimension of the latent space. Defaults to 100.
    """
    
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
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
    
    def train_generator(self, optimizer_G, adversarial_loss, discriminator, real_labels, batch_size, device):
        """
        Train the generator.
        
        Parameters
        ----------
        optimizer_G : torch.optim.Optimizer
            Optimizer for the generator.
        adversarial_loss : torch.nn.Module
            Adversarial loss function.
        discriminator : torch.nn.Module
            Discriminator.
        real_labels : torch.Tensor
            Tensor of shape (batch_size, 1) containing the real labels.
        batch_size : int
            Batch size.
        device : torch.device
            Device to use for training.
        """
        
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, self.latent_dim).to(device)
        gen_imgs = self(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), real_labels)
        g_loss.backward()
        optimizer_G.step()
        return g_loss

class Discriminator(nn.Module):
    """Discriminator class for the GAN."""
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, bn=True):
            """
            Discriminator block. 
            Consists of a convolutional layer, a leaky ReLU activation, and a dropout layer.
            """
            block = [
                nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1), 
                nn.LeakyReLU(0.2, inplace=True), 
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 128 // 2**4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())


    def forward(self, img):
        """
        Forward pass for the discriminator.
        
        Parameters
        ----------
        img : torch.Tensor
            Input tensor of shape (batch_size, 3, 128, 128).
        
        Returns
        -------
        validity : torch.Tensor
            Output tensor of shape (batch_size, 1).
        """
        
        out = self.model(img)
        out = out.view(out.size(0), -1)
        validity = self.adv_layer(out)
        return validity
    
    def train_discriminator(self, optimizer_D, adversarial_loss, real_imgs, fake_imgs, real_labels, fake_labels):
        """
        Train the discriminator.
        
        Parameters
        ----------
        optimizer_D : torch.optim.Optimizer
            Optimizer for the discriminator.
        adversarial_loss : torch.nn.Module
            Adversarial loss function.
        real_imgs : torch.Tensor
            Tensor of shape (batch_size, 3, 128, 128) containing the real images.
        fake_imgs : torch.Tensor
            Tensor of shape (batch_size, 3, 128, 128) containing the fake images.
        real_labels : torch.Tensor
            Tensor of shape (batch_size, 1) containing the real labels.
        fake_labels : torch.Tensor
            Tensor of shape (batch_size, 1) containing the fake labels.
        """
        
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(self(real_imgs), real_labels)
        fake_loss = adversarial_loss(self(fake_imgs.detach()), fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        return d_loss
                    
class GAN(AbstractModel):
    """
    GAN class that inherits from our AbstractModel.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to use.
    lr : float
        Learning rate.
    latent_dim : int
        Dimension of the latent space.
    batch_size : int
        Batch size.
    device : torch.device
        Device to use for training.
    """
    
    def __init__(self, dataset_name="ffhq_raw", lr=0.0002, latent_dim=DEFAULT_LATENT_DIM, batch_size=64, device="cpu"):
        # Initialize the abstract class
        super().__init__(dataset_name, batch_size)

        # GAN-specific attributes
        self.generator = Generator(latent_dim).to(device)
        self.discriminator = Discriminator().to(device)
        self.latent_dim = latent_dim
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.adversarial_loss = nn.BCELoss()

        # Apply the weights initialization
        self.generator.apply(self.weights_init_normal)
        self.discriminator.apply(self.weights_init_normal)

    @staticmethod
    def weights_init_normal(m):
        """
        Initialize weights with a normal distribution.
        
        Parameters
        ----------
        m : torch.nn.Module
            Module to initialize. In practice, this is either the generator or the discriminator.
        """
        
        classname = m.__class__.__name__
        if classname.find("Linear") != -1: 
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
                
    def train(self, num_epochs, device, log_interval=100, save_interval=1, test_batches_limit=None, checkpoint_path=None):
        """
        Main training loop.
        
        Parameters
        ----------
        num_epochs : int
            Number of epochs to train for.
        device : torch.device
            Device to use for training.
        log_interval : int
            Number of batches to wait before logging training progress. Defaults to 100. Can be None.
        save_interval : int
            Number of epochs to wait before saving the models and generated images. Defaults to 10.
        test_batches_limit : int
            Number of batches to use for testing. Defaults to None.
        checkpoint_path : str
            Path to a checkpoint to load. Defaults to None.
        """
        
        dataloaders = {"train": self.train_loader, "test": self.test_loader}

        start_epoch = 0
        if checkpoint_path:
            start_epoch = self.load_checkpoint(checkpoint_path, device)
            print(f"Resuming training from epoch {start_epoch}...")
        
        for epoch in range(start_epoch, num_epochs):
            for phase in ["train", "test"]:
                if phase == "train":
                    self.generator.train()
                else:
                    self.generator.eval()

                running_loss = 0.0
                
                data_iter = tqdm(
                    enumerate(dataloaders[phase]), 
                    total=len(dataloaders[phase]), 
                    desc=f"Epoch {epoch+1}/{num_epochs} [{phase}]"
                )

                for i, imgs in data_iter:
                    imgs = imgs.to(device)
                    batch_size = imgs.size(0)
                    real_labels = torch.ones(batch_size, 1).to(device)
                    fake_labels = torch.zeros(batch_size, 1).to(device)

                    if phase == "train":
                        # Zero the parameter gradients
                        self.optimizer_G.zero_grad()
                        self.optimizer_D.zero_grad()

                        # Forward pass, backward pass, and optimize
                        g_loss, d_loss = self.perform_train_step(imgs, real_labels, fake_labels, batch_size, device)
                        running_loss += g_loss.item() + d_loss.item()

                        if log_interval is not None and i % log_interval == 0:
                            self.log_training(epoch, num_epochs, i, len(dataloaders[phase]), g_loss, d_loss)
                        
                        # Early stopping for pytest
                        if test_batches_limit is not None and i == test_batches_limit:
                            break
                    
                    else:  # phase == "test"
                        with torch.no_grad():
                            g_loss, d_loss = self.perform_validation_step(imgs, real_labels, fake_labels, batch_size, device)
                            running_loss += g_loss.item() + d_loss.item()

                epoch_loss = running_loss / len(dataloaders[phase])
                self.epoch_losses[phase].append(epoch_loss)
            
            if (epoch + 1) % save_interval == 0:
                self.save_models(epoch)
                self.save_checkpoint(epoch)
                self.save_generated_images(epoch, device)
    
    def perform_train_step(self, real_imgs, real_labels, fake_labels, batch_size, device):
        """
        One step for the training phase.
        
        Parameters
        ----------
        real_imgs : torch.Tensor
            Tensor of shape (batch_size, 3, 128, 128) containing the real images.
        real_labels : torch.Tensor
            Tensor of shape (batch_size, 1) containing the real labels.
        fake_labels : torch.Tensor
            Tensor of shape (batch_size, 1) containing the fake labels.
        batch_size : int
            Batch size.
        device : torch.device
            Device to use for training.
        """
        
        # Generate fake images
        z = torch.randn(batch_size, self.latent_dim).to(device)
        fake_imgs = self.generator(z)

        # Train Generator
        self.optimizer_G.zero_grad()
        g_loss = self.adversarial_loss(self.discriminator(fake_imgs), real_labels)
        g_loss.backward()
        self.optimizer_G.step()

        # Train Discriminator
        self.optimizer_D.zero_grad()
        real_loss = self.adversarial_loss(self.discriminator(real_imgs), real_labels)
        fake_loss = self.adversarial_loss(self.discriminator(fake_imgs.detach()), fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()

        return g_loss, d_loss

    def perform_validation_step(self, real_imgs, real_labels, fake_labels, batch_size, device):
        """
        One step for the validation phase.
        
        Parameters
        ----------
        real_imgs : torch.Tensor
            Tensor of shape (batch_size, 3, 128, 128) containing the real images.
        real_labels : torch.Tensor
            Tensor of shape (batch_size, 1) containing the real labels.
        fake_labels : torch.Tensor
            Tensor of shape (batch_size, 1) containing the fake labels.
        batch_size : int
            Batch size.
        device : torch.device
            Device to use for training.
        """
        
        # Generate fake images
        z = torch.randn(batch_size, self.latent_dim).to(device)
        fake_imgs = self.generator(z)
        
        # Compute generator loss
        g_loss = self.adversarial_loss(self.discriminator(fake_imgs), real_labels)

        # Compute discriminator loss
        fake_loss = self.adversarial_loss(self.discriminator(fake_imgs), fake_labels)
        real_loss = self.adversarial_loss(self.discriminator(real_imgs), real_labels)
        d_loss = (real_loss + fake_loss) / 2
        
        return g_loss, d_loss

                
    def log_training(self, epoch, num_epochs, batch_idx, total_batches, d_loss, g_loss):
        """
        Log training progress.
        
        Parameters
        ----------
        epoch : int
            Current epoch.
        num_epochs : int
            Total number of epochs.
        batch_idx : int
            Current batch index.
        total_batches : int
            Total number of batches.
        d_loss : torch.Tensor
            Discriminator loss.
        g_loss : torch.Tensor
            Generator loss.
        """
        
        print(f"Dataset {self.dataset_name} - Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx}/{total_batches} "
            f"Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")

    def save_models(self, epoch, save_dir="outputs/models"):
        """
        Save only the models.
        
        Parameters
        ----------
        epoch : int
            Current epoch.
        save_dir : str
            Directory to save the models to.
        """
        save_folder = f"{save_dir}_{self.dataset_name}"
        os.makedirs(save_folder, exist_ok=True)
        
        torch.save(self.generator.state_dict(), f"{save_folder}/generator_epoch_{epoch+1}.pth")
        torch.save(self.discriminator.state_dict(), f"{save_folder}/discriminator_epoch_{epoch+1}.pth")

    def save_generated_images(self, epoch, device, save_dir="outputs/generated_images"):
        """
        Save generated images.
        
        Parameters
        ----------
        epoch : int
            Current epoch.
        device : torch.device
            Device to use for training.
        save_dir : str
            Directory to save the images to.
        """
        
        save_folder = f"{save_dir}_{self.dataset_name}"
        os.makedirs(save_folder, exist_ok=True)
        z = torch.randn(64, self.latent_dim).to(device)  # Generate random latent vectors
        fake_images = self.generator(z).detach().cpu()
        save_image(fake_images, f"{save_folder}/epoch_{epoch+1}.png", nrow=8, normalize=True)
        
    def save_checkpoint(self, epoch, save_dir="outputs/checkpoints"):
        """
        Save a checkpoint of the current state. This includes the models, optimizers, and losses.
        
        Parameters
        ----------
        epoch : int
            Current epoch.
        save_dir : str
            Directory to save the checkpoint to.
        """
        
        save_folder = f"{save_dir}_{self.dataset_name}"
        os.makedirs(save_folder, exist_ok=True)
        checkpoint_path = os.path.join(save_folder, f"checkpoint_epoch_{epoch}.pth")
        
        checkpoint = {
            "epoch": epoch + 1, # Since we want to start from the next epoch
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_state_dict": self.optimizer_D.state_dict(),
            "losses": self.epoch_losses
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path, device):
        """
        Load the checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        self.optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        self.epoch_losses = checkpoint["losses"]
        
        return checkpoint["epoch"]