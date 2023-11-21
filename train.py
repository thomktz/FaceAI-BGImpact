import os
import json
import argparse
import torch
from models import DCGAN, StyleGAN

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Deep Learning Model Training")
    
    # Model-specific arguments
    parser.add_argument("--model", type=str, required=True, choices=["DCGAN", "StyleGAN"], help="Type of model to train")
    parser.add_argument("--dataset", type=str, required=True, choices=["ffhq_raw", "ffhq_blur", "ffhq_grey"], help="Name of the dataset to use")
    parser.add_argument("--latent-dim", type=int, help="Dimension of the latent space")
    parser.add_argument("--config-path", type=str, default=None, help="Path to a custom JSON configuration file")
    
    # Training arguments
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--loss", type=str, choices=["wgan", "wgan-gp", "basic"], default="wgan-gp", help="Learning rate decay")
    parser.add_argument("--c", type=float, help="Clipping parameter for Discriminator weights")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--num-epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--save-interval", type=int, help="Number of epochs to wait before saving models and images")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to a checkpoint file to resume training. Has priority over --checkpoint-epoch")
    parser.add_argument("--checkpoint-epoch", type=int, default=None, help="Epoch number of the checkpoint to resume training from")
    return parser.parse_args()

def load_default_config(model):
    """Load the default configuration JSON for the specified model type."""
    default_config_path = os.path.join("configs", f"default_{model.lower()}_config.json")
    with open(default_config_path, "r") as default_config_file:
        default_config = json.load(default_config_file)
    return default_config

def main(args):
    """Main parsing script."""
    
    default_config = load_default_config(args.model)
    
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
    if args.model.lower() == "dcgan":
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
        train_config = dict(
            num_epochs=config["num_epochs"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            device=device,
            save_interval=config["save_interval"],
        )
    elif args.model.lower() == "stylegan":
        if checkpoint_path is None:
            model = StyleGAN(
                dataset_name=args.dataset,
                latent_dim=config["latent_dim"],
                w_dim=config["w_dim"],
                style_layers=config["style_layers"],
                device=device,
            )
        else:
            model = StyleGAN.from_checkpoint(
                checkpoint_path=checkpoint_path,
                device=device,
            )
        train_config = dict(
            lr=config["lr"],
            loss=config["loss"],
            batch_size=config["batch_size"],
            device=device,
            save_interval=config["save_interval"],
            level_epochs={
                int(k): int(v) 
                for (k, v) in config["level_epochs"].items()
            },
            transition_ratio=config["transition_ratio"],
            c=config["c"],
        )

    else:
        raise ValueError("Invalid model type")

    model.train(**train_config)

if __name__ == "__main__":
    args = parse_args()
    main(args)
