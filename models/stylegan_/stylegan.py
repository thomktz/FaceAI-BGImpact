import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from pytorch_gan_metrics import get_fid

from models.abstract_model import AbstractModel
from models.data_loader import get_dataloader, denormalize_imagenet
from models.utils import weights_init
from models.stylegan_.generator import Generator
from models.stylegan_.discriminator import Discriminator
from models.stylegan_.utils import compute_gradient_penalty

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

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.optimizer_G_config = {}
        self.optimizer_D_config = {}

    def train_init(self, lr):
        """
        Initialize the training process.
        
        Parameters
        ----------
        lr : float
            Learning rate.
        batch_size : int
            Batch size.
        """
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        # Load optimizer states if available
        if self.optimizer_D_config:
            self.optimizer_D.load_state_dict(self.optimizer_D_config)
        if self.optimizer_G_config:
            self.optimizer_G.load_state_dict(self.optimizer_G_config)

    def train(self, lr, batch_size, lambda_gp, device, save_interval, level_epochs, transition_ratio):
        """
        Main training loop for StyleGAN.
        
        Parameters
        ----------
        lr : float
            Learning rate.
        batch_size : int
            Batch size.
        lambda_gp : float
            Gradient penalty lambda hyperparameter.
        device : torch.device
            Device to use for training.
        save_interval : int
            Number of epochs to wait before saving models and images.
        level_epochs : dict
            Dictionary mapping resolution levels to the number of epochs to train at that level.
        """
        self.train_init(lr)
        total_steps = sum(level_epochs.values())
        current_step = 0

        for level, level_steps in level_epochs.items():
            current_resolution = 4 * 2 ** level
            
            transition_steps = int(level_steps * transition_ratio)
            
            print(f"Training level {level} with resolution {current_resolution} for {level_steps} steps")
            
            self.loader = get_dataloader(self.dataset_name, batch_size, resolution=current_resolution)
            for level_step in range(level_steps):
                current_step += 1
                if level_step < transition_steps and level > 0:
                    # Transition phase
                    alpha = (level_step+0.5) / (transition_steps+0.5)
                else:
                    # Stabilization phase
                    alpha = 1.0
        
                self._train_one_epoch(level_step, level_steps, current_step, total_steps, level, alpha, lambda_gp, device, save_interval)

    def _train_one_epoch(self, level_step, level_steps, current_step, total_steps, level, alpha, lambda_gp, device, save_interval):
        # Training loop for one epoch
        self.generator.train()
        running_loss = 0.0

        data_iter = tqdm(enumerate(self.loader), total=len(self.loader), desc=f"Level {level} Epoch {level_step+1}/{level_steps} total {current_step}/{total_steps} alpha {alpha:.2f}")

        for i, imgs in data_iter:
            if i%50 == 0:
                self.generate_images(i, device, level, alpha)
            imgs = imgs.to(device)

            g_loss, d_loss = self.perform_train_step(imgs, lambda_gp, device, level, alpha)
            running_loss += g_loss + d_loss

        self.generate_images(current_step, device, level, alpha)
        if current_step % save_interval == 0:
            self.save_checkpoint(current_step, level, alpha, device)

    def perform_train_step(self, real_imgs, lambda_gp, device, current_level, alpha, drift=0.001):
        """
        Perform a single training step, including forward and backward passes for both
        the generator and discriminator.

        Parameters
        ----------
        real_imgs : torch.Tensor
            Real images batch.
        lambda_gp : float
            Gradient penalty lambda hyperparameter.
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

        z = torch.randn(current_batch_size, self.latent_dim, device=device)
        fake_imgs = self.generator(z, current_level, alpha)

        # Calculate discriminator loss on real images
        real_scores = self.discriminator(real_imgs, current_level, alpha)

        # Calculate discriminator loss on fake images
        fake_scores = self.discriminator(fake_imgs.detach(), current_level, alpha)

        d_loss = (
            torch.mean(fake_scores)
            - torch.mean(real_scores)
            + (drift * torch.mean(real_scores ** 2))
        )

        # calculate the WGAN-GP (gradient penalty)
        gp = compute_gradient_penalty(self.discriminator, real_imgs.data, fake_imgs.data, current_level, alpha, device)
        d_loss += lambda_gp * gp

        # Update discriminator
        self.optimizer_D.zero_grad()
        d_loss.backward()  # retain_graph is not needed here
        self.optimizer_D.step()

        # Calculate generator loss
        g_loss = -torch.mean(self.discriminator(fake_imgs, current_level, alpha))
        
        # Update generator
        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()

        return g_loss.item(), d_loss.item()
    
    
    def save_checkpoint(self, current_step, current_level, alpha, save_dir="outputs/StyleGAN_checkpoints"):
        """
        Save a checkpoint of the current state, including models, optimizers, and training parameters.

        Parameters
        ----------
        current_step : int
            Current training step.
        current_level : int
            Current training level.
        alpha : float
            Current alpha value for blending resolutions.
        device : torch.device
            Device to use for training.
        save_dir : str
            Directory to save the checkpoint to.
        """
        save_folder = os.path.join(save_dir, self.dataset_name)
        os.makedirs(save_folder, exist_ok=True)

        checkpoint_path = os.path.join(save_folder, f"checkpoint_step_{current_step}_level_{current_level}.pth")
        checkpoint = {
            "current_step": current_step,
            "current_level": current_level,
            "alpha": alpha,
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
        current_level = checkpoint["current_level"]
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
        instance.current_step = checkpoint["current_step"]
        instance.current_level = current_level
        instance.alpha = alpha

        return instance

    def generate_images(self, epoch, device, level, alpha, save_dir="outputs/StyleGAN_images"):
        """
        Save generated images.
        
        Parameters
        ----------
        epoch : int
            Current epoch.
        device : torch.device
            Device to use for training.
        level : int
            Current resolution level.
        alpha : float
            Current alpha value for blending resolutions.
        save_dir : str
            Directory to save the images to.
        """
        
        save_folder = self.get_save_dir(save_dir)
        os.makedirs(save_folder, exist_ok=True)
        
        z = torch.randn(64, self.latent_dim).to(device)
        
        fake_images = self.generator(z, level, alpha).detach().cpu()
        denormalized_images = denormalize_imagenet(fake_images)
        save_image(denormalized_images, f"{save_folder}/epoch_{epoch}.png", nrow=8, normalize=False)