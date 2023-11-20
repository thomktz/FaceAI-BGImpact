import torch.nn as nn
import torch.nn.init as init

def weights_init(std=0.02):
    def f(m):
        """
        Initialize the weights of a neural network module.
        
        Parameters:
        ----------
        m : nn.Module
            A module of the neural network.
        """
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            # Apply weight initialization to convolutional and linear layers
            init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            # Apply weight initialization to batch normalization layers
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        
    return f