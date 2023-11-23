"""
DCDCGAN implementation.

https://arxiv.org/pdf/1511.06434.pdf]
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchvision.utils import save_image
from pytorch_gan_metrics import get_fid

from faceai_bgimpact.data_processing.paths import data_folder
from faceai_bgimpact.models.abstract_model import AbstractModel
from faceai_bgimpact.models.data_loader import get_dataloader, denormalize_imagenet
from faceai_bgimpact.models.utils import weights_init

from faceai_bgimpact.models.dcgan_.generator import Generator
from faceai_bgimpact.models.dcgan_.discriminator import Discriminator


class DCGAN(AbstractModel):
    """
    DCGAN class that inherits from our AbstractModel.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to use.
    latent_dim : int
        Dimension of the latent space.
    device : torch.device
        Device to use for training.
    """
    
    def __init__(self, dataset_name, latent_dim, device):
        # Initialize the abstract class
        super().__init__(dataset_name)

        # DCGAN-specific attributes
        self.generator = Generator(latent_dim).to(device)
        self.discriminator = Discriminator().to(device)
        self.latent_dim = latent_dim

        # Apply the weights initialization
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        
        self.start_epoch = 0
        self.optimizer_G_config = {}
        self.optimizer_D_config = {}
    
    def train_init(self, lr, batch_size):
        """Initialize the training parameters and optimizer."""
        
        _, self.loader = get_dataloader(self.dataset_name, batch_size)
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.adversarial_loss = nn.BCELoss()
        
        if self.optimizer_D_config:
            print("Using loaded optimizer_D state...")
            self.optimizer_D.load_state_dict(self.optimizer_D_config)
        if self.optimizer_G_config:
            print("Using loaded optimizer_G state...")
            self.optimizer_G.load_state_dict(self.optimizer_G_config)
        
    
    def train(self, num_epochs, lr, batch_size, device, save_interval):
        """
        Main training loop.
        
        Parameters
        ----------
        num_epochs : int
            Number of epochs to train for.
        lr : float
            Learning rate.
        batch_size : int
            Batch size.
        device : torch.device
            Device to use for training.
        save_interval : int
            Number of epochs to wait before saving the models and generated images. Defaults to 10.
        """
        # Initialize training parameters and optimizer
        self.train_init(lr, batch_size)
        
        for epoch in range(self.start_epoch, num_epochs):
            self.generator.train()
            running_loss = 0.0
            
            data_iter = tqdm(
                enumerate(self.loader), 
                total=len(self.loader), 
                desc=f"Epoch {epoch+1}/{num_epochs}"
            )

            for i, imgs in data_iter:
                imgs = imgs.to(device)
                batch_size = imgs.size(0)
                real_labels = torch.ones(batch_size, 1).to(device)
                fake_labels = torch.zeros(batch_size, 1).to(device)

                # Zero the parameter gradients
                # TODO: Why do we need to zero the gradients here?
                # We're doing it twice
                self.optimizer_G.zero_grad()
                self.optimizer_D.zero_grad()

                # Forward pass, backward pass, and optimize
                g_loss, d_loss = self.perform_train_step(imgs, real_labels, fake_labels, batch_size, device)
                
                running_loss += g_loss.item() + d_loss.item()

            epoch_loss = running_loss / len(self.loader)
            self.epoch_losses["train"].append(epoch_loss)
            
            
            
            if ((epoch + 1) % save_interval == 0) or (epoch <= 3):
                print(f"Saving checkpoint at epoch {epoch+1}...")
                self.save_checkpoint(epoch)
                
            self.generate_images(epoch, device)
            
            # Evaluate the FID score, and log it as 'test' loss
            # FID needs at least 2048 images to compare the final average pooling features
            # C.f. https://github.com/mseitzer/pytorch-fid
            # We use 2048 + batch_size to ensure that ((2048 + batch_size) // batch_size) * batch_size > 2048
            fid_score = self.calculate_fid(2048+batch_size, batch_size, device)
            self.epoch_losses["test"].append(fid_score)
            print(f"FID: {fid_score:.2f}")
    
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

    def calculate_fid(self, num_images, batch_size, device):
        """
        Calculate the FID score.
        
        Parameters
        ----------
        device : torch.device
            Device to use for calculation.
        
        Returns
        -------
        fid_score : float
            Computed FID score.
        """
        self.generator.eval()
        images = []
        with torch.no_grad():
            for _ in range(num_images // batch_size):
                z = torch.randn(batch_size, self.latent_dim).to(device)
                images.append(self.generator(z).detach().cpu())
        
        self.generator.train()
        imgs = torch.cat(images, dim=0) 
        denormalized_imgs = denormalize_imagenet(imgs)
        
        stats_path = f"{data_folder}/{self.dataset_name}_statistics.npz"
        return get_fid(denormalized_imgs, stats_path)

    def generate_images(self, epoch, device, save_dir="outputs/generated_images"):
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
        
        save_folder = self.get_save_dir(save_dir)
        os.makedirs(save_folder, exist_ok=True)
        
        z = torch.randn(64, self.latent_dim).to(device)
        
        fake_images = self.generator(z).detach().cpu()
        denormalized_images = denormalize_imagenet(fake_images)
        save_image(denormalized_images, f"{save_folder}/epoch_{epoch+1}.png", nrow=8, normalize=False)
        
    def generate_one_image(self, device, save_folder, filename):
        """Generate one image and save it to the directory."""
        
        os.makedirs(save_folder, exist_ok=True)
        
        z = torch.randn(1, self.latent_dim).to(device)
        
        fake_image = self.generator(z).detach().cpu()
        denormalized_image = denormalize_imagenet(fake_image)
        save_image(denormalized_image, os.path.join(save_folder, filename), normalize=False)
        
        
        
    def save_checkpoint(self, epoch, save_dir="outputs/checkpoints"):
        """
        Save a checkpoint of the current state. This includes the models, optimizers, losses,
        and training parameters.
        
        Parameters
        ----------
        epoch : int
            Current epoch.
        save_dir : str
            Directory to save the checkpoint to.
        """
        save_folder = self.get_save_dir(save_dir)
        os.makedirs(save_folder, exist_ok=True)
        
        checkpoint_path = os.path.join(save_folder, f"checkpoint_epoch_{epoch+1}.pth")
        checkpoint = {
            "epoch": epoch + 1,  # Since we want to start from the next epoch
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_state_dict": self.optimizer_D.state_dict(),
            "losses": self.epoch_losses,
            "dataset_name": self.dataset_name,
            "latent_dim": self.latent_dim,
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device):
        """
        Create a DCGAN instance from a checkpoint file.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.
        device : torch.device
            Device to use for training.

        Returns
        -------
        instance : DCGAN
            A DCGAN instance with loaded state.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract necessary components from checkpoint
        dataset_name = checkpoint.get("dataset_name", "ffhq_raw")
        latent_dim = checkpoint.get("latent_dim", 100)

        # Create a new DCGAN instance
        instance = cls(dataset_name, latent_dim, device)

        # Load the state into the instance
        instance.generator.load_state_dict(checkpoint["generator_state_dict"])
        instance.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        instance.epoch_losses = checkpoint["losses"]
        instance.start_epoch = checkpoint["epoch"]
        
        # Load the optimizer state
        instance.optimizer_G_config = checkpoint["optimizer_G_state_dict"]
        instance.optimizer_D_config = checkpoint["optimizer_D_state_dict"]

        return instance
