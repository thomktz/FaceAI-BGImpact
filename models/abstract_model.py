import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from models.data_loader import get_dataloaders

class AbstractModel(ABC):
    """
    An abstract class that represents a model in our framework.
    Subclasses are required to implement the defined abstract methods.
    """
    
    def __init__(self, dataset_name, batch_size):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.train_loader, self.test_loader = get_dataloaders(dataset_name, batch_size)
        self.epoch_losses = {"train": [], "test": []}
    
    @abstractmethod
    def train(self, num_epochs, device, log_interval, save_interval, test_batches_limit=None):
        """
        Run the training loop for the model.
        """
        pass
    
    @abstractmethod
    def save_models(self, epoch):
        """
        Save the model parameters.
        """
        pass
    
    @abstractmethod
    def save_generated_images(self, epoch, device):
        """
        Save images generated by the model.
        """
        pass

    @staticmethod
    @abstractmethod
    def weights_init_normal(m):
        """
        Initialize weights of the model with a normal distribution.
        """
        pass