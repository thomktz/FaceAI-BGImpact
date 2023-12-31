from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class GANLoss(ABC):
    """Abstract base class for GAN Losses."""

    def __init__(self, G, D):
        self.G = G
        self.D = D

    @abstractmethod
    def d_loss(self, real_images, fake_images, level, alpha):
        """Discriminator loss."""
        pass

    @abstractmethod
    def g_loss(self, real_images, fake_images, level, alpha):
        """Generator loss."""
        pass


class WGAN(GANLoss):
    """Wasserstein GAN (WGAN) loss."""

    def __init__(self, G, D):
        super().__init__(G, D)

    def d_loss(self, real_images, fake_images, level, alpha):
        """Discriminator loss."""
        real_scores = self.D(real_images, level, alpha)
        fake_scores = self.D(fake_images, level, alpha)
        loss = torch.mean(fake_scores) - torch.mean(real_scores)
        return loss

    def g_loss(self, _, fake_images, level, alpha):
        """Generator loss."""
        fake_scores = self.D(fake_images, level, alpha)
        return -torch.mean(fake_scores)


class WGAN_GP(GANLoss):
    """
    Wasserstein GAN with Gradient Penalty (WGAN-GP) loss.

    Parameters:
    ----------
    G : torch.nn.Module
        Generator network.
    D : torch.nn.Module
        Discriminator network.
    lambda_gp : float
        Gradient penalty coefficient.
    drift : float
        Drift coefficient.
    """

    def __init__(self, G, D, lambda_gp=10, drift=0.001):
        super().__init__(G, D)
        self.drift = drift
        self.lambda_gp = lambda_gp

    def d_loss(self, real_images, fake_images, level, alpha):
        """Discriminator loss."""
        real_scores = self.D(real_images, level, alpha)
        fake_scores = self.D(fake_images, level, alpha)

        loss = torch.mean(fake_scores) - torch.mean(real_scores) + (self.drift * torch.mean(real_scores**2))

        # calculate the WGAN-GP (gradient penalty)
        gp = self._gradient_penalty(real_images, fake_images, level, alpha, real_images.device)
        loss += gp

        return loss

    def g_loss(self, _, fake_images, level, alpha):
        """Generator loss."""
        fake_scores = self.D(fake_images, level, alpha)
        return -torch.mean(fake_scores)

    def _gradient_penalty(self, real_images, fake_images, level, alpha, device):
        """Calculates the gradient penalty loss for WGAN GP."""
        batch_size = real_images.shape[0]

        # generate random epsilon
        epsilon = torch.rand((batch_size, 1, 1, 1)).to(device)

        # create the merge of both real and fake samples
        merged = epsilon * real_images + ((1 - epsilon) * fake_images)
        merged.requires_grad_(True)

        # forward pass
        op = self.D(merged, level, alpha)

        # perform backward pass from op to merged for obtaining the gradients
        gradient = torch.autograd.grad(
            outputs=op,
            inputs=merged,
            grad_outputs=torch.ones_like(op),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradient = gradient.view(gradient.shape[0], -1)

        return self.lambda_gp * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()


class R1Regularization(GANLoss):
    """
    R1 Regularization for the Discriminator.

    Parameters:
    ----------
    G : torch.nn.Module
        Generator network.
    D : torch.nn.Module
        Discriminator network.
    lambda_r1 : float
        Coefficient for the R1 regularization term.
    """

    def __init__(self, G, D, lambda_r1=10):
        super().__init__(G, D)
        self.lambda_r1 = lambda_r1

    def d_loss(self, real_images, fake_images, level, alpha):
        """Discriminator loss with R1 regularization."""
        real_scores = torch.mean(self.D(real_images, level, alpha))
        fake_scores = torch.mean(self.D(fake_images, level, alpha))

        # Standard GAN loss
        loss = fake_scores - real_scores

        # R1 regularization
        r1_penalty = self._r1_penalty(real_images, level, alpha)
        loss += r1_penalty
        return loss

    def g_loss(self, _, fake_images, level, alpha):
        """Generator loss."""
        fake_scores = torch.mean(self.D(fake_images, level, alpha))
        return -fake_scores

    def _r1_penalty(self, real_images, level, alpha):
        """Calculates the R1 regularization term."""
        # Requires grad enables automatic differentiation for real_images
        real_images.requires_grad_(True)

        # Forward pass
        real_scores = self.D(real_images, level, alpha)

        # Calculate gradients
        real_gradients = torch.autograd.grad(
            outputs=real_scores.sum(), inputs=real_images, create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Compute the R1 penalty
        r1_penalty = self.lambda_r1 * real_gradients.pow(2).view(real_gradients.shape[0], -1).sum(1).mean()

        return r1_penalty


class BasicGANLoss(GANLoss):
    """Basic GAN Loss using Binary Cross-Entropy (BCE) loss."""

    def __init__(self, G, D):
        super().__init__(G, D)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def d_loss(self, real_images, fake_images, level, alpha):
        """Discriminator loss."""
        # Real images should be classified as real (label=1)
        real_scores = self.D(real_images, level, alpha)
        real_loss = self.loss_fn(real_scores, torch.ones_like(real_scores))

        # Fake images should be classified as fake (label=0)
        fake_scores = self.D(fake_images, level, alpha)
        fake_loss = self.loss_fn(fake_scores, torch.zeros_like(fake_scores))

        # Total discriminator loss
        loss = real_loss + fake_loss

        return loss

    def g_loss(self, _, fake_images, level, alpha):
        """Generator loss."""
        # Generator aims to have fake images classified as real (label=1)
        fake_scores = self.D(fake_images, level, alpha)
        loss = self.loss_fn(fake_scores, torch.ones_like(fake_scores))

        return loss
