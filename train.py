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
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to a checkpoint file to resume training. Has priority over --checkpoint-epoch")
    parser.add_argument("--checkpoint-epoch", type=int, default=None, help="Epoch number of the checkpoint to resume training from")
    parser.add_argument("--to-drive", default=False, action='store_true', help="Flag indicating whether to save outputs to Google Drive")
    parser.add_argument("--drive-path", type=str, default="AML/", help="Subdirectory path in Google Drive to save the outputs")
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
    
    # Set checkpoint path if resuming from a checkpoint
    checkpoint_path = None
    if args.checkpoint_epoch is not None:
        checkpoint_dir = "outputs/checkpoints_{}".format(args.dataset_name)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{args.checkpoint_epoch}.pth")
        print(f"Resuming from checkpoint: {checkpoint_path}")
        
    if args.checkpoint_path is not None:
        checkpoint_path = args.checkpoint_path
        print(f"Resuming from checkpoint: {checkpoint_path}")

    # Initialize the appropriate model
    if args.model_type == "gan":
        model = GAN(
            dataset_name=config["dataset_name"],
            lr=config["lr"],
            latent_dim=config["latent_dim"],
            batch_size=config["batch_size"],
            device=device,
            drive_path=args.drive_path,
            to_drive=args.to_drive
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
        checkpoint_path=checkpoint_path,
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)
