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
    parser.add_argument("--frame-rate", type=int, default=30, help="Frame rate for the video")
    parser.add_argument("--compress", action="store_true", help="Enable lossless compression for the video")
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
    font = ImageFont.truetype("Nirmala.ttf", 40)
    text_position = (30, image.height - 60)  # adjust based on your image size
    draw.text(text_position, text, font=font, fill=(255, 255, 255))  # white text
    return image

def get_iter(path):
    """Gets the integer after "iter" in the path."""
    return int(path.split("iter_")[1].split("_")[0])

def create_video(image_folder, output_video, frame_rate, compress):
    """Create a video from images."""
    images = [img for img in sorted(os.listdir(image_folder), key=get_iter) if img.endswith(".png")]
    if not images:
        raise ValueError("No images found in the specified folder.")

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    
    codec = 'XVID' if compress else 'mp4v'
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*codec), frame_rate, (width, height))

    for image_name in tqdm(images):
        iter_, level, epoch, alpha = map(float, image_name[:-4].split('_')[1:]) 
        resolution = 4 * 2 ** int(level)
        img_path = os.path.join(image_folder, image_name)

        text = f"Level: {int(level)} ({resolution}x{resolution}), Epoch: {int(epoch)}, Alpha: {alpha:.2f}"
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
        output_video=f"outputs/{args.model}_video_{args.dataset}.mp4",
        frame_rate=args.frame_rate,
        compress=args.compress,
    )
