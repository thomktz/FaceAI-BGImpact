import os
import json
import argparse
import torch
import time
from models.gan import GAN

def parse_args():
    """Parse command-line arguments for image generation."""
    parser = argparse.ArgumentParser(description="Generate Images using Trained Models")
    parser.add_argument("--model-type", type=str, required=True, choices=["gan"], help="Type of model to use for generation")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--num-images", type=int, default=10, help="Number of images to generate")
    parser.add_argument("--output-dir", type=str, default="generated_images", help="Directory to save generated images")
    return parser.parse_args()

def load_model(checkpoint_path, device):
    """Load the GAN model from a checkpoint."""
    start_time = time.time()
    model = GAN.from_checkpoint(checkpoint_path, device)
    elapsed_time = time.time() - start_time
    print(f"Took {elapsed_time:.2f} seconds to load model")
    return model

def generate_images(model, num_images, output_dir, device):
    """Generate and save images using the model."""
    start_time = time.time()
    for i in range(num_images):
        model.generate_one_image(save_folder=output_dir, filename=f"image_{i}.png", device=device)

    elapsed_time = time.time() - start_time
    print(f"Took {elapsed_time:.2f} seconds to generate {num_images} images")

def main(args):
    """Main script for generating images from a trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.checkpoint_path, device)
    generate_images(model, args.num_images, args.output_dir, device)

if __name__ == "__main__":
    args = parse_args()
    main(args)
