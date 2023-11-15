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
    parser.add_argument("--model-type", type=str, required=True, choices=["gan", "vae"], help="Type of model to train")
    parser.add_argument("--dataset-name", type=str, required=True, choices=["ffhq_raw", "ffhq_blur", "ffhq_grey"], help="Name of the dataset to use")
    parser.add_argument("--config-path", type=str, default=None, help="Path to a custom JSON configuration file")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, help="Dimension of the latent space")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--num-epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--log-interval", type=int, help="Number of batches to wait before logging training progress")
    parser.add_argument("--save-interval", type=int, help="Number of epochs to wait before saving models and images")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to a checkpoint file to resume training")
    return parser.parse_args()

def load_default_config(model_type):
    default_config_path = os.path.join("configs", f"default_{model_type}_config.json")
    with open(default_config_path, "r") as default_config_file:
        default_config = json.load(default_config_file)
    return default_config

def main(args):
    # Load the default configuration based on the model type
    default_config = load_default_config(args.model_type)

    # Load the custom configuration from the provided JSON file if specified
    if args.config_path:
        with open(args.config_path, "r") as custom_config_file:
            custom_config = json.load(custom_config_file)
    else:
        custom_config = {}

    # Merge the default and custom configurations, giving priority to custom settings
    config = {**default_config, **custom_config}

    # Override specific settings with command-line arguments, if provided
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    # Set the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the appropriate model
    if args.model_type == "gan":
        model = GAN(
            dataset_name=config["dataset_name"],
            lr=config["lr"],
            latent_dim=config["latent_dim"],
            batch_size=config["batch_size"],
            device=device,
        )
    else:
        raise ValueError("Invalid model type")

    # Training loop (using the selected model)
    model.train(
        num_epochs=config["num_epochs"],
        device=device,
        log_interval=config["log_interval"],
        save_interval=config["save_interval"],
        test_batches_limit=None,
        checkpoint_path=args.checkpoint_path,
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)
