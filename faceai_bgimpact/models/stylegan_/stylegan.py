import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.transforms import Resize
from pytorch_gan_metrics import get_fid

from faceai_bgimpact.models.abstract_model import AbstractModel
from faceai_bgimpact.models.data_loader import get_dataloader, denormalize_image
from faceai_bgimpact.models.utils import pairwise_euclidean_distance
from faceai_bgimpact.models.stylegan_.generator import Generator
from faceai_bgimpact.models.stylegan_.discriminator import Discriminator
from faceai_bgimpact.models.stylegan_.loss import WGAN_GP, BasicGANLoss, WGAN

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
        self.w_dim = w_dim
        self.style_layers = style_layers

        self.optimizer_G_config = {}
        self.optimizer_D_config = {}
        
        self.latent_vector = torch.randn(64, self.latent_dim)

    def train_init(self, glr, mlr, dlr, loss):
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
        loss : str
            Loss function to use for training.
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
        
        self.level = 0
        self.resolution = 4
        self.alpha = 1
        self.epoch_total = None
        
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
            
        # Check if loading from a checkpoint
        if hasattr(self, 'current_epochs'):
            # Check if checkpoint was the last epoch of the level
            if self.current_epochs[self.level] == (level_epochs[self.level]["training"] + level_epochs[self.level]["transition"]):
                # If so, move to the next level
                self.level += 1
                self.resolution = 4 * (2 ** self.level)
                self.alpha = 0
                self.dataset.update_resolution(self.resolution)
                self.current_epochs[self.level] = 0
            
            start_level = self.level
            start_epoch = self.current_epochs[self.level]
        else:
            start_level = 0
            start_epoch = 0
            self.current_epochs = {level: 0 for level in level_epochs.keys()}
            self.train_init(
                glr=glr, 
                mlr=mlr, 
                dlr=dlr,
                loss=loss
            )
        
        self.dataset, self.loader = get_dataloader(self.dataset_name, batch_size, shuffle=True, resolution=self.resolution, alpha=1.0)
        
        for level in range(start_level, max(level_epochs.keys()) + 1):
            self.level = level
            self.resolution = 4 * (2 ** level)
            self.dataset.update_resolution(self.resolution)

            # Calculate total epochs for this level from configuration
            total_level_epochs = level_epochs[level]["transition"] + level_epochs[level]["training"]
            start_epoch_for_level = start_epoch if level == start_level else 0

            for epoch in range(start_epoch_for_level, total_level_epochs):
                self.current_epochs[level] = epoch
                self.epoch_total = sum(self.current_epochs.values())
                self._train_one_epoch(level_epochs[level], epoch, image_interval, device)

                # Save checkpoint
                if (self.epoch_total + 1) % save_interval == 0:
                    self.current_epochs[level] = epoch + 1
                    self.save_checkpoint(self.epoch_total + 1, self.current_epochs)

            # Update for next level
            if level < max(level_epochs.keys()):
                self.alpha = 0.0  # Reset alpha for the next level

                
            
    def _train_one_epoch(self, level_config, epoch, image_interval, device):
        """Training loop for one epoch."""
        
        # Determine if we are in the transition phase
        is_transition_phase = self.alpha < 1.0

        # Calculate alpha step if in transition phase
        alpha_step = 1.0 / (len(self.loader) * level_config["transition"]) if is_transition_phase else 0
        
        
        def tqdm_description(self, epoch, total_epochs, g_loss=0, d_loss=0):
            
            return (
                f"Lvl {' ' * (3 - len(str(self.resolution))) * 2}{self.level} ({self.resolution}x{self.resolution}) "
                + f"Epoch {epoch+1}/{total_epochs} "
                + f"α={self.alpha:.2f} "
                + f"GL={g_loss:.3f} DL={d_loss:.3f} "
                + f"d={self.real_distance:.1f}/{self.fake_distance:.1f}"
            )
            
        total_epochs = level_config["transition"] + level_config["training"]
        epoch_iter = tqdm(enumerate(self.loader), total=len(self.loader), desc=tqdm_description(self, epoch, total_epochs))

        for i, imgs in epoch_iter:
            # Update alpha
            self.alpha = min(self.alpha + alpha_step, 1.0)
            
            # Update dataset alpha
            self.dataset.update_alpha(self.alpha)
            
            # Train on batch
            imgs = imgs.to(device)
            g_loss, d_loss = self.perform_train_step(imgs, device, self.level, self.alpha)

            # Update tqdm description
            epoch_iter.desc = tqdm_description(self, epoch, total_epochs, g_loss, d_loss)
            
            if i % image_interval == 0:
                epoch_total = sum(self.current_epochs.values())
                iter_ = (epoch_total * len(self.loader)) + i
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
    
    def calculate_completed_epochs(self, alpha, level_config, epochs_before):
        """
        Calculate the number of epochs completed at the current level based on alpha.

        Parameters:
        ----------
        alpha : float
            Current value of alpha.
        level_config : dict
            Configuration for the current level.
        epochs_before : int
            Number of epochs completed before the current level.

        Returns:
        -------
        int
            Number of completed epochs at the current level.
        """
        if self.epoch_total is not None:
            return self.epoch_total - epochs_before
        if alpha < 1.0:
            # If alpha is not yet 1, we are still in the transition phase
            completed_fraction = alpha * level_config["transition"]
        else:
            # If alpha is 1, the transition phase is complete
            completed_fraction = level_config["transition"]

        return int(completed_fraction)
    
    def save_checkpoint(self, epoch_total, current_epochs, save_dir="outputs/StyleGAN_checkpoints"):
        """
        Save a checkpoint of the current state, including models, optimizers, and training parameters.

        Parameters
        ----------
        epoch_total : int
            Current training step.
        current_epochs : int
            Number of epochs completed at the current level.
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
            "current_epochs": current_epochs,
            "level": self.level,
            "alpha": self.alpha,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_state_dict": self.optimizer_D.state_dict(),
            "w_dim": self.w_dim,
            "latent_dim": self.latent_dim,
            "style_layers": self.style_layers,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")


    @classmethod
    def from_checkpoint(cls, dataset_name, checkpoint_path, loss, device, latent_dim=None, w_dim=None, style_layers=None):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract necessary components from checkpoint
        level = checkpoint["level"]
        alpha = checkpoint["alpha"]
        epoch_total = checkpoint["epoch_total"]
        current_epochs = checkpoint["current_epochs"]
        w_dim = checkpoint.get("w_dim", w_dim)
        latent_dim = checkpoint.get("latent_dim", latent_dim)
        style_layers = checkpoint.get("style_layers", style_layers)

        # Create a new StyleGAN instance
        instance = cls(dataset_name, latent_dim, w_dim, style_layers, device)
        
        # Initialize the instance with 0 learning rates
        instance.train_init(0, 0, 0, loss)

        # Load the state into the instance
        instance.generator.load_state_dict(checkpoint["generator_state_dict"])
        instance.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        instance.optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        instance.optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])

        # Set current step, level, and epoch tracking
        instance.level = level
        instance.alpha = alpha
        instance.resolution = 4 * (2 ** level)
        instance.epoch_total = epoch_total
        instance.current_epochs = current_epochs
        
        print("Loaded state:")
        print(f"  Level: {instance.level}")
        print(f"  Alpha: {instance.alpha}")
        print(f"  Epoch total: {instance.epoch_total}")
        print(f"  Current epochs: {instance.current_epochs}")

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

            real_images = []
            for _ in range(min(len(self.dataset), 64)):
                real_images.append(self.dataset[_])
            real_images = torch.stack(real_images).to(device)
            
            # Resize if necessary
            if real_images.size(-1) != 128:
                upscaler = Resize((128, 128), interpolation=0)  # 0 corresponds to nearest-neighbor
                real_images = upscaler(real_images)
            
            # Denormalize and save real images
            real_images = denormalize_image(real_images.cpu())
            save_image(real_images, f"{save_folder}/real_{iter_}_{self.level}_{epoch}_{self.alpha:.2f}.png", nrow=8, normalize=False)


            # Generate images
            fake_images = self.generator(z, self.level, self.alpha).detach().cpu()
            
            # Check if upscaling is needed
            current_size = fake_images.size(-1)
            if current_size != 128:
                upscaler = Resize((128, 128), interpolation=0)  # 0 corresponds to nearest-neighbor
                fake_images = upscaler(fake_images)

            # Denormalize and save images
            denormalized_images = denormalize_image(fake_images)
            save_image(denormalized_images, f"{save_folder}/fake_{iter_}_{self.level}_{epoch}_{self.alpha:.2f}.png", nrow=8, normalize=False)