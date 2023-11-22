import argparse
import json
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def parse_args():
    """Parse command-line arguments for video creation."""
    parser = argparse.ArgumentParser(description="Create a video from GAN training frames")

    parser.add_argument("--model", type=str, required=True, choices=["DCGAN", "StyleGAN"], help="Type of model")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset used")
    parser.add_argument("--output-video", type=str, default="training_video.mp4", help="Path for the output video file")
    parser.add_argument("--frame-rate", type=int, default=30, help="Frame rate for the video")
    parser.add_argument("--transition-ratio", type=float, default=0.1, help="Transition ratio used during training")
    return parser.parse_args()

def load_model_config(model, dataset):
    """Load the configuration JSON for the specified model and dataset."""
    config_path = os.path.join("configs", f"default_{model.lower()}_config.json")
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config

def add_text_to_image(image, text):
    """Add text to an image."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # or specify a font file
    text_position = (10, image.height - 20)  # adjust based on your image size
    draw.text(text_position, text, font=font, fill=(255, 255, 255))  # white text
    return image

def infer_alpha(epoch, level_epochs, transition_ratio):
    """Infer alpha value based on the current epoch."""
    accumulated_epochs = 0
    for level, level_steps in level_epochs.items():
        if accumulated_epochs + level_steps > epoch:
            # Current level is found
            transition_steps = int(level_steps * transition_ratio)
            epoch_in_level = epoch - accumulated_epochs
            if epoch_in_level < transition_steps and int(level) > 0:
                alpha = (epoch_in_level + 0.5) / (transition_steps + 0.5)
            else:
                alpha = 1.0
            return alpha
        accumulated_epochs += level_steps
    return 1.0  # Default alpha for epochs beyond the specified levels

def create_video(image_folder, output_video, frame_rate, level_epochs, transition_ratio):
    """Create a video from images."""
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    if not images:
        raise ValueError("No images found in the specified folder.")

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    for image_name in tqdm(images):
        batch, epoch = map(int, image_name.split('.')[0].split('_')[1:])  # Split filename to get epoch
        alpha = infer_alpha(epoch, level_epochs, transition_ratio)
        img_path = os.path.join(image_folder, image_name)

        text = f"Epoch: {epoch}, Batch: {batch}, Alpha: {alpha:.2f}"
        img = Image.open(img_path)
        img_with_text = add_text_to_image(img, text)
        img_with_text.save(img_path)

        video_frame = cv2.imread(img_path)
        video.write(video_frame)

    video.release()

if __name__ == "__main__":
    args = parse_args()
    config = load_model_config(args.model, args.dataset)

    image_folder = f"outputs/{args.model}_images_{args.dataset}"
    create_video(
        image_folder=image_folder,
        output_video=args.output_video,
        frame_rate=args.frame_rate,
        level_epochs=config["level_epochs"],
        transition_ratio=args.transition_ratio
    )