import os
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from importlib import resources

# Assuming the font file is still accessible the same way
with resources.path('faceai_bgimpact', 'Nirmala.ttf') as font_path:
    font = ImageFont.truetype(str(font_path), size=40)

def add_text_to_image(image, text):
    """Add text to an image."""
    draw = ImageDraw.Draw(image)
    text_position = (30, image.height - 60)
    draw.text(text_position, text, font=font, fill=(255, 255, 255))
    return image

def create_video(image_folder, output_video, frame_rate):
    """Create a video from images using imageio."""
    filenames = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")], 
                        key=lambda x: (int(x.split("_")[1]), x.split("_")[0]))

    if not filenames:
        raise ValueError("No images found in the specified folder.")

    output_format = output_video.split(".")[-1]

    with imageio.get_writer(output_video, fps=frame_rate, format=output_format, codec='libx265', quality=10) as writer:
        for i in range(0, len(filenames), 2): 
            real_img_name = filenames[i]
            fake_img_name = filenames[i + 1]

            iter_, level, epoch, alpha = map(float, real_img_name[:-4].split('_')[1:5])
            resolution = 4 * 2 ** int(level)
            real_img_path = os.path.join(image_folder, real_img_name)
            fake_img_path = os.path.join(image_folder, fake_img_name)

            text = f"Level: {int(level)} ({resolution}x{resolution}), Epoch: {int(epoch)}, Alpha: {alpha:.2f}"
            
            # Open images
            real_img = Image.open(real_img_path)
            fake_img = Image.open(fake_img_path)

            # Combine images
            combined_img = Image.new('RGB', (real_img.width + fake_img.width, real_img.height))
            combined_img.paste(real_img, (0, 0))
            combined_img.paste(fake_img, (real_img.width, 0))

            combined_img_with_text = add_text_to_image(combined_img, text)

            # Convert PIL Image to numpy array
            img_array = np.array(combined_img_with_text)
            
            writer.append_data(img_array)

def create_video_function(model, dataset, frame_rate):
    """Create a video from images using imageio."""
    image_folder = f"outputs/{model}_images_{dataset}"
    output_video = f"outputs/{model}_video_{dataset}.mp4"
    create_video(image_folder, output_video, frame_rate)

