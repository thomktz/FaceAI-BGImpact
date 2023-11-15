import os
import json
import argparse
import torch
from datetime import datetime
from tqdm import tqdm
from torchvision.utils import save_image
from models.gan import GAN
from models.data_loader import get_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Model Training")
    # Only GAN works for now, VAE later
    parser.add_argument("--model_type", type=str, required=True, choices=["gan", "vae"], help="Type of model to train")
    parser.add_argument("--config_path", type=str, default=None, help="Path to a custom JSON configuration file")
    return parser.parse_args()

def load_default_config(model_type):
    default_config_path = os.path.join("configs", f"default_{model_type}_config.json")
    with open(default_config_path, "r") as default_config_file:
        default_config = json.load(default_config_file)
    return default_config

def main(args):
    # Load the default configuration based on the model type
    default_config = load_default_config(args.model_type)

    # Merge the default configuration with the custom JSON configuration file if provided
    if args.config_path:
        with open(args.config_path, "r") as custom_config_file:
            custom_config = json.load(custom_config_file)
        config = {**default_config, **custom_config}
    else:
        config = default_config

    # Set the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the appropriate model
    if args.model_type == "gan":
        model = GAN(
            dataset=config["dataset_name"],
            lr=config["lr"],
            latent_dim=config["latent_dim"],
            batch_size=config["batch_size"]
        )
    elif args.model_type == "vae":
        raise NotImplementedError("VAE not implemented yet")
    else:
        raise ValueError("Invalid model type")

    # Training loop (using the selected model)
    model.train(
        num_epochs=config["num_epochs"],
        device=device,
        log_interval=config["log_interval"],
        save_interval=config["save_interval"],
        test_batches_limit=config["test_batches_limit"]
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)
