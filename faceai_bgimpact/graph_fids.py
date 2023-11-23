"""Loads a checkpoint and graphs the loss over time."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from faceai_bgimpact.models.dcgan import DCGAN

def graph_fids(checkpoint_path, save_dir):
    
    model = DCGAN.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
    )
    
    fids = model.epoch_losses["test"]
    
    # Create the plot
    plt.plot(fids)
    plt.title("FID vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("FID")
    plt.savefig(os.path.join(save_dir, "fid_vs_epoch.png"))
    plt.close()
    
