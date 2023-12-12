import os
import torch

from pytorch_gan_metrics import get_fid
from pytorch_gan_metrics.utils import calc_and_save_stats

from faceai_bgimpact.data_processing.paths import data_folder
from faceai_bgimpact.models.data_loader import get_dataloader, denormalize_image
from faceai_bgimpact.models.abstract_model import AbstractModel

from provae import ProVAE as BaseProVAE


class ProVAE(BaseProVAE, AbstractModel):
    """
    Progressive-growing Variational Autoencoder.

    Inherits from:
    -----------
    BaseProVAE : class
        The base ProVAE class.
        https://github.com/thomktz/ProVAE
    AbstractModel : class
        The abstract model class.

    Parameters:
    ----------
    dataset : str
        The dataset to use.
    latent_dim : int
        The dimension of the latent space.
    config : dict
        The configuration of the model.
    """

    def __init__(self, dataset, latent_dim, config):
        BaseProVAE.__init__(self, latent_dim, config, "cpu")
        AbstractModel.__init__(self, dataset)

        self.optimizer_config = {}
        self.latent_vector = torch.randn(64, self.latent_dim)
        self.fids = {"level": [], "epoch": [], "fid": []}

    def train_init(self, lr):
        """
        Initialize the training process.

        Parameters:
        ----------
        lr : float
            The learning rate.
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.0, 0.99), eps=1e-8)

        self.level = 0
        self.resolution = self.config[self.level]["resolution"]
        self.alpha = 1.0
        self.epoch_total = None

    def train(self, lr, device, save_interval, image_interval):
        """
        Train the model.

        Parameters:
        ----------
        lr : float
            The learning rate.
        device : str
            The device to use.
        save_interval : int
            The number of epochs between saving checkpoints.
        image_interval : int
            The number of epochs between saving images.
        """
        if hasattr(self, "current_epochs"):
            if self.current_epochs[self.level] == (
                self.config[self.level]["transition_epochs"] + self.config[self.level]["stabilization_epochs"]
            ):
                # If so, move to the next level
                self.level += 1
                self.resolution = self.config[self.level]["resolution"]
                self.alpha = 0
                self.current_epochs[self.level] = 0

            start_level = self.level
            start_epoch = self.current_epochs[self.level]
        else:
            start_level = 0
            start_epoch = 0
            self.current_epochs = {level: 0 for level in range(self.config)}
            self.train_init(lr=lr)

        for level in range(start_level, len(range(self.config))):
            # Set the resolution and alpha
            self.level = level
            self.resolution = self.config[level]["resolution"]
            self.batch_size = self.config[level]["batch_size"]

            self.dataset, self.loader = get_dataloader(
                self.dataset_name,
                self.batch_size,
                shuffle=True,
                resolution=self.resolution,
                alpha=self.alpha,
            )

            # Calculate total epochs for this level from configuration
            total_level_epochs = self.config[level]["transition_epochs"] + self.config[level]["stabilization_epochs"]
            start_epoch_for_level = start_epoch if level == start_level else 0

            # Computing FID stats for current level
            self.make_fid_stats(device)

            # Epoch loop for current level
            for epoch in range(start_epoch_for_level, total_level_epochs):
                self.current_epochs[level] = epoch
                self.epoch_total = sum(self.current_epochs.values())
                self._train_one_epoch(epoch, image_interval, device)

                # Calculate FID score of epoch
                if self.alpha == 1.0:
                    self.calculate_fid(2048 + self.batch_size, self.batch_size, device)

                # Save checkpoint
                if (self.epoch_total + 1) % save_interval == 0:
                    self.current_epochs[level] = epoch + 1
                    self.save_checkpoint(self.epoch_total + 1, self.current_epochs)

            # Update for next level
            if level < len(self.config) - 1:
                self.alpha = 0.0  # Reset alpha for the next level

    def make_fid_stats(self, device, save_dir="outputs/ProVAE_fid_stats"):
        """
        Calculate and save FID stats for the dataset.

        Parameters
        ----------
        device : torch.device
            Device to use for calculation.
        save_dir : str
            Directory to save the stats to.
        """
        # If device is CPU, ignore and skip
        print(device)
        if str(device) == "cpu":
            return

        dataset_folder = f"{data_folder}/{self.dataset_name}"
        save_file = self.get_save_dir(save_dir) + f"{self.resolution}.npz"

        # If the stats file already exists, skip
        if os.path.exists(save_file):
            return

        print("Generating FID stats for resolution", self.resolution, "...")
        calc_and_save_stats(
            dataset_folder,
            save_file,
            batch_size=64,
            img_size=self.resolution,
            use_torch=True,
            num_workers=os.cpu_count(),
        )

    def calculate_fid(self, num_images, batch_size, device, stats_dir="outputs/ProVAE_fid_stats"):
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
        if str(device) == "cpu":
            return
        self.generator.eval()
        images = []
        with torch.no_grad():
            for _ in range(num_images // batch_size):
                z = torch.randn(batch_size, self.latent_dim).to(device)
                images.append(denormalize_image(self.generator(z, self.level, self.alpha).detach()).cpu())

        self.generator.train()
        imgs = torch.cat(images, dim=0)

        stats_file = self.get_save_dir(stats_dir) + f"{self.resolution}.npz"
        fid = get_fid(imgs, stats_file)
        self.fids["level"].append(self.level)
        self.fids["epoch"].append(self.epoch_total)
        self.fids["fid"].append(fid)
        print("Level", self.level, "Epoch", self.epoch_total, "FID", fid)
