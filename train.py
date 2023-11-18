import os
import json
import argparse
import torch
from models.dcgan import DCGAN

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Deep Learning Model Training")
    
    # Model-specific arguments
    parser.add_argument("--model-type", type=str, required=True, choices=["DCGAN", "VAE"], help="Type of model to train")
    parser.add_argument("--dataset", type=str, required=True, choices=["ffhq_raw", "ffhq_blur", "ffhq_grey"], help="Name of the dataset to use")
    parser.add_argument("--latent-dim", type=int, help="Dimension of the latent space")
    parser.add_argument("--config-path", type=str, default=None, help="Path to a custom JSON configuration file")
    
    # Training arguments
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--num-epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--save-interval", type=int, help="Number of epochs to wait before saving models and images")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to a checkpoint file to resume training. Has priority over --checkpoint-epoch")
    parser.add_argument("--checkpoint-epoch", type=int, default=None, help="Epoch number of the checkpoint to resume training from")
    return parser.parse_args()

def load_default_config(model_type):
    """Load the default configuration JSON for the specified model type."""
    default_config_path = os.path.join("configs", f"default_{model_type.lower()}_config.json")
    with open(default_config_path, "r") as default_config_file:
        default_config = json.load(default_config_file)
    return default_config

def main(args):
    """Main parsing script."""
    
    default_config = load_default_config(args.model_type)
    
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set checkpoint path if resuming from a checkpoint
    checkpoint_path = None
        
    if args.checkpoint_epoch is not None:
        checkpoint_dir = f"outputs/checkpoints_{args.dataset}"
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{args.checkpoint_epoch}.pth")
        
    elif args.checkpoint_path is not None:
        checkpoint_path = args.checkpoint_path

    if args.model_type == "DCGAN":
        if checkpoint_path is None:
            model = DCGAN(
                dataset_name=args.dataset,
                latent_dim=config["latent_dim"],
                device=device,
            )
        else:
            model = DCGAN.from_checkpoint(
                checkpoint_path=checkpoint_path,
                device=device,
            )

    else:
        raise ValueError("Invalid model type")

    model.train(
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        device=device,
        save_interval=config["save_interval"],
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)
