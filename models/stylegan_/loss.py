import torch
from abc import ABC, abstractmethod

class GANLoss(ABC):
    def __init__(self, G, D):
        self.G = G
        self.D = D

    @abstractmethod
    def d_loss(self, real_images, fake_images, level, alpha):
        pass

    @abstractmethod
    def g_loss(self, real_images, fake_images, level, alpha):
        pass
    
class WGAN_GP(GANLoss):
    def __init__(self, G, D, lambda_gp=10.0, drift=0.001):
        super().__init__(G, D)
        self.drift = drift
        self.lambda_gp = lambda_gp
    
    def d_loss(self, real_images, fake_images, level, alpha):
        real_scores = self.D(real_images, level, alpha)
        fake_scores = self.D(fake_images, level, alpha)
        loss = (
            torch.mean(fake_scores)
            - torch.mean(real_scores)
            + (self.drift * torch.mean(real_scores ** 2))
        )

        # calculate the WGAN-GP (gradient penalty)
        gp = self._gradient_penalty(
            real_images, fake_images, level, alpha, real_images.device
        )
        loss += gp

        return loss
    
    def g_loss(self, _, fake_images, level, alpha):
        fake_scores = self.D(fake_images, level, alpha)
        return -torch.mean(fake_scores)
        
    def _gradient_penalty(self, real_images, fake_images, level, alpha, device):
        """Calculates the gradient penalty loss for WGAN GP"""
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