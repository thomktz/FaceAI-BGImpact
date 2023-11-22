import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.transforms import Resize
from pytorch_gan_metrics import get_fid

from models.abstract_model import AbstractModel
from models.data_loader import get_dataloader, denormalize_imagenet
from models.utils import pairwise_euclidean_distance
from models.stylegan_.generator import Generator
from models.stylegan_.discriminator import Discriminator
from models.stylegan_.loss import WGAN_GP, BasicGANLoss, WGAN

class StyleGAN(AbstractModel):
    """
    StyleGAN class inheriting from AbstractModel.

    Parameters
    ----------
    dataset : str
        Name of the dataset to use.
    latent_dim : int
        Dimension of the latent space.
    w_dim : int
        Dimension of the W space.
    style_layers : int
        Number of layers in the style mapping network.
    device : torch.device
        Device to use for training.
    """

    def __init__(self, dataset_name, latent_dim, w_dim, style_layers, device):
        super().__init__(dataset_name)
        
        self.generator = Generator(latent_dim, w_dim, style_layers).to(device)
        self.discriminator = Discriminator().to(device)
        self.latent_dim = latent_dim

        self.optimizer_G_config = {}
        self.optimizer_D_config = {}
        
        self.latent_vector = torch.randn(64, self.latent_dim)

    def train_init(self, glr, mlr, dlr):
        """
        Initialize the training process.
        
        Parameters
        ----------
        glr : float
            Learning rate for the generator.
        mlr : float
            Learning rate for the mapping network.
            The StyleGAN paper uses mlr = 0.01 * glr.
        dlr : float
            Learning rate for the discriminator.
        """
        
        self.optimizer_G = optim.Adam(
            [
                {"params": self.generator.mapping.parameters(), "lr": mlr},
                {"params": self.generator.synthesis.parameters()}
            ], 
            lr=glr, 
            betas=(0.0, 0.99), 
            eps=1e-8
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=dlr, 
            betas=(0.0, 0.99), 
            eps=1e-8
        )
        
        self.real_distance = 0.0
        self.fake_distance = 0.0

        # Load optimizer states if available
        if self.optimizer_D_config:
            self.optimizer_D.load_state_dict(self.optimizer_D_config)
        if self.optimizer_G_config:
            self.optimizer_G.load_state_dict(self.optimizer_G_config)

    def train(self, glr, mlr, dlr, batch_size, device, save_interval, image_interval, level_epochs, loss):
        """
        Main training loop for StyleGAN.
        
        Parameters
        ----------
        lr : float
            Learning rate.
        batch_size : int
            Batch size.
        device : torch.device
            Device to use for training.
        save_interval : int
            Number of epochs to wait before saving models and images.
        level_epochs : dict
            Dictionary mapping resolution levels to the number of epochs to train at that level.
        transition_ratio : float
            Ratio of the total number of epochs to use for the transition phase.
        loss : str
            Loss function to use for training.
            ["wgan-gp", "wgan", "basic"]
        """
        self.loss = {
            "wgan": WGAN,
            "wgan-gp": WGAN_GP,
            "basic": BasicGANLoss
        }.get(
            loss.lower().replace("-", "_"),
            WGAN_GP
        )(
            self.generator, 
            self.discriminator
        )
        self.train_init(
            glr=glr, 
            mlr=mlr, 
            dlr=dlr
        )
        
        self.level = 0
        self.resolution = 4
        self.alpha = 1
        self.dataset, self.loader = get_dataloader(self.dataset_name, batch_size, shuffle=True, resolution=self.resolution, alpha=1.0)
        n_batches = len(self.loader)
        
        for level, level_config in level_epochs.items():
            self.level = level
            epochs_before = sum([cfg["transition"] + cfg["training"] for (lvl, cfg) in level_epochs.items() if lvl < level])
            
            # Update resolution for the level
            self.dataset.update_resolution(self.resolution)

            total_level_epochs = level_config["transition"] + level_config["training"]
            alpha_step = 1.0 / (level_config["transition"] * n_batches) if level_config["transition"] > 0 else 0

            for epoch in range(total_level_epochs):
                self._train_one_epoch(alpha_step, epoch, total_level_epochs, device, image_interval)

                # Save checkpoint
                epoch_total = epochs_before + epoch + 1
                if epoch_total % save_interval == 0:
                    self.save_checkpoint(epoch_total)
            
            # Move to the next level
            if self.level < max(level_epochs.keys()):
                self.level += 1
                self.resolution *= 2
                self.alpha = 0.0
            
            
                  
                

    def _train_one_epoch(self, alpha_step, epoch, total_epochs, device, image_interval):
        """Training loop for one epoch."""
        
        def tqdm_description(self, epoch, total_epochs, g_loss=0, d_loss=0):
            return f"Lvl {self.level} ({self.resolution}x{self.resolution}) Epoch {epoch+1}/{total_epochs} α={self.alpha:.2f} GL={g_loss:.3f} DL={d_loss:.3f} d={self.real_distance:.1f}/{self.fake_distance:.1f}"
    
        epoch_iter = tqdm(enumerate(self.loader), total=len(self.loader), desc=tqdm_description(self, epoch, total_epochs))
        for i, imgs in epoch_iter:
            # Update alpha
            self.alpha = min(self.alpha + alpha_step, 1.0)
            self.dataset.update_alpha(self.alpha)
            
            # Train on batch
            imgs = imgs.to(device)
            g_loss, d_loss = self.perform_train_step(imgs, device, self.level, self.alpha)

            # Update tqdm description
            epoch_iter.desc = tqdm_description(self, epoch, total_epochs, g_loss, d_loss)
            
            if i % image_interval == 0:
                iter_ = (epoch * len(self.dataset)) + i
                self.generate_images(iter_, epoch, device, latent_vector=self.latent_vector)
            

    def perform_train_step(self, real_imgs, device, current_level, alpha):
        """
        Perform a single training step, including forward and backward passes for both
        the generator and discriminator.

        Parameters
        ----------
        real_imgs : torch.Tensor
            Real images batch.
        device : torch.device
            Device on which to perform computations.
        current_level : int
            Current resolution level of the model.
        alpha : float
            Blending factor for the progressive growing.

        Returns
        -------
        g_loss : torch.Tensor
            Generator loss for the step.
        d_loss : torch.Tensor
            Discriminator loss for the step.
        """
        # On the last batch of the epoch, the number of images may be less than the batch size
        current_batch_size = real_imgs.size(0)

        # Reset gradients
        self.optimizer_D.zero_grad()
        self.optimizer_G.zero_grad()
        
        # Train discriminator
        z = torch.randn(current_batch_size, self.latent_dim, device=device)
        detached_fake_imgs = self.generator(z, current_level, alpha).detach()
        d_loss = self.loss.d_loss(real_imgs, detached_fake_imgs, current_level, alpha)
        d_loss.backward()
        self.optimizer_D.step()

        # Train generator
        z = torch.randn(current_batch_size, self.latent_dim, device=device)
        fake_imgs = self.generator(z, current_level, alpha)
        g_loss = self.loss.g_loss(None, fake_imgs, current_level, alpha)
        g_loss.backward()
        self.optimizer_G.step()

        # Compute distances
        self.real_distance = pairwise_euclidean_distance(real_imgs)
        self.fake_distance = pairwise_euclidean_distance(fake_imgs)
        
        return g_loss.item(), d_loss.item()
    
    
    def save_checkpoint(self, epoch_total, save_dir="outputs/StyleGAN_checkpoints"):
        """
        Save a checkpoint of the current state, including models, optimizers, and training parameters.

        Parameters
        ----------
        epoch_total : int
            Current training step.
        device : torch.device
            Device to use for training.
        save_dir : str
            Directory to save the checkpoint to.
        """
        save_folder = self.get_save_dir(save_dir)
        os.makedirs(save_folder, exist_ok=True)

        checkpoint_path = os.path.join(save_folder, f"step_{epoch_total}_{self.level}_{self.alpha:.2f}.pth")
        checkpoint = {
            "epoch_total": epoch_total,
            "level": self.level,
            "alpha": self.alpha,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_state_dict": self.optimizer_D.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device, w_dim, style_layers):
        """
        Create a StyleGAN instance from a checkpoint file.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.
        device : torch.device
            Device to use for training.
        w_dim : int
            Dimension of the W space in StyleGAN.
        style_layers : int
            Number of layers in the style mapping network.

        Returns
        -------
        instance : StyleGAN
            A StyleGAN instance with loaded state.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract necessary components from checkpoint
        dataset_name = checkpoint.get("dataset_name", "default_dataset")
        latent_dim = checkpoint.get("latent_dim", 100)  # Default value if not in checkpoint
        level = checkpoint["level"]
        alpha = checkpoint["alpha"]

        # Create a new StyleGAN instance
        instance = cls(dataset_name, latent_dim, w_dim, style_layers, device)

        # Load the state into the instance
        instance.generator.load_state_dict(checkpoint["generator_state_dict"])
        instance.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        instance.optimizer_G = optim.Adam(instance.generator.parameters())
        instance.optimizer_D = optim.Adam(instance.discriminator.parameters())
        instance.optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        instance.optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])

        # Set current step and level
        instance.level = level
        instance.alpha = alpha

        return instance

    def generate_images(self, iter_, epoch, device, save_dir="outputs/StyleGAN_images", latent_vector=None):
        """
        Save generated images.
        
        Parameters
        ----------
        iter_ : int
            Current iteration.
        device : torch.device
            Device to use for training.
        save_dir : str
            Directory to save the images to.
        latent_vector : torch.Tensor, optional
            Latent vector to generate images from. If None, generates a random vector.
        """
        with torch.no_grad():
            save_folder = self.get_save_dir(save_dir)
            os.makedirs(save_folder, exist_ok=True)
            
            # Use provided latent vector or generate a new one
            if latent_vector is None:
                z = torch.randn(64, self.latent_dim).to(device)
            else:
                z = latent_vector.to(device)

            # Generate images
            fake_images = self.generator(z, self.level, self.alpha).detach().cpu()

            # Check if upscaling is needed
            current_size = fake_images.size(-1)
            if current_size != 128:
                upscaler = Resize((128, 128), interpolation=0)  # 0 corresponds to nearest-neighbor
                fake_images = upscaler(fake_images)

            # Denormalize and save images
            denormalized_images = denormalize_imagenet(fake_images)
            save_image(denormalized_images, f"{save_folder}/iter_{iter_}_{self.level}_{epoch}_{self.alpha:.2f}.png", nrow=8, normalize=False)