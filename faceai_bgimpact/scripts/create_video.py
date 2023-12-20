import os
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from importlib import resources

# Assuming the font file is still accessible the same way
with resources.path("faceai_bgimpact", "Nirmala.ttf") as font_path:
    font = ImageFont.truetype(str(font_path), size=50)


def add_text_to_image(image, text):
    """Add text to an image."""
    draw = ImageDraw.Draw(image)
    text_position = (30, image.height - 70)
    draw.text(text_position, text, font=font, fill=(255, 255, 255))
    return image


def create_video(image_folder, output_video, frame_rate, skip_frames, model_type):
    """Create a video from images using imageio."""
    filenames = sorted(
        [img for img in os.listdir(image_folder) if "fake" in img], key=lambda x: int(x.split("_")[2].split(".")[0])
    )

    if not filenames:
        raise ValueError("No images found in the specified folder.")

    output_format = output_video.split(".")[-1]
    real_image_path = os.path.join(image_folder, "real.png") if model_type == "VAE" else None

    with imageio.get_writer(output_video, fps=frame_rate, format=output_format, codec="libx265", quality=10) as writer:
        for filename in filenames[:: skip_frames + 1]:
            fake_image_path = os.path.join(image_folder, filename)

            if model_type.lower() == "stylegan":
                iter_, level, epoch, alpha = map(float, filename[:-4].split("_")[1:5])
                resolution = 4 * 2 ** int(level)
                real_image_path = os.path.join(image_folder, f"real_{iter_}_{level}_{epoch}_{alpha:.2f}.png")
                text = f"Level: {int(level)} ({resolution}x{resolution}), Epoch: {int(epoch)}, Alpha: {alpha:.2f}"
            elif model_type.lower() == "vae":
                epoch = int(filename.split("_")[2].split(".")[0])
                text = f"Epoch: {epoch}"

            # Open images
            fake_img = Image.open(fake_image_path)
            real_img = Image.open(real_image_path) if real_image_path and os.path.exists(real_image_path) else fake_img

            # Combine images
            combined_img = Image.new("RGB", (real_img.width + fake_img.width, real_img.height))
            combined_img.paste(real_img, (0, 0))
            combined_img.paste(fake_img, (real_img.width, 0))

            combined_img_with_text = add_text_to_image(combined_img, text)

            # Convert PIL Image to numpy array
            img_array = np.array(combined_img_with_text)

            writer.append_data(img_array)


def create_video_function(model, dataset, frame_rate, skip_frames):
    """
    Create a video from images using imageio.

    Parameters:
    ----------
    model : str
        Type of model.
    dataset : str
        Name of the dataset used.
    frame_rate : int
        Frame rate for the video.
    skip_frames : int
        Number of frames to skip between each frame.
    """
    image_folder = f"outputs/{model}_images_{dataset}"
    output_video = f"outputs/{model}_video_{dataset}.mp4"
    create_video(image_folder, output_video, frame_rate, skip_frames, model)
