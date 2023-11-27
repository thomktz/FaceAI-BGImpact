import os
import base64
import torch
from io import BytesIO
from time import time
from flask import request
from flask_restx import Resource
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image

from .namespace import api
from .load_trained_model import stylegan
from .models import image_model, latent_vector_model, style_vector_model
from faceai_bgimpact.models.data_loader import denormalize_image

NUM_IMAGES = 16
stored_w_vectors = None


to_pil_image = ToPILImage()

def time_function(func, *args, **kwargs):
    start_time = time()
    result = func(*args, **kwargs)
    end_time = time()
    elapsed_time = end_time - start_time
    return result, elapsed_time

def normalized_tensor_to_b64(tensor):
    # Generate image and denormalize
    dn_tensor = denormalize_image(tensor.clamp(-1, 1))
    
    filename = "temp/" + str(time()) + ".jpg"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save_image(dn_tensor, filename, nrow=4, normalize=False)
    
    # Convert to PIL Image
    image = to_pil_image(dn_tensor.squeeze(0))
    
    # Convert the PIL Image to a BytesIO object
    buffered = BytesIO()
    image.save(buffered, format="JPEG")

    # Encode the image as a base64 string
    return base64.b64encode(buffered.getvalue()).decode()
    

@api.route("/randomize-latents")
class RandomizeLatents(Resource):
    @api.marshal_with(image_model)
    def get(self):
        global stored_w_vectors
        z = torch.randn(NUM_IMAGES, stylegan.latent_dim).to("cpu")
        with torch.no_grad():
            w_vectors, _ = time_function(stylegan.generator.mapping, z)
        stored_w_vectors = w_vectors

        with torch.no_grad():
            tensors, time_ = time_function(stylegan.generator.predict_from_style, stored_w_vectors, stylegan.level, stylegan.alpha, False)

        images_b64 = [normalized_tensor_to_b64(tensor.unsqueeze(0)) for tensor in tensors]

        return {"images": images_b64, "time": time_}, 200


@api.route("/apply-style-sliders")
class ApplyStyleSliders(Resource):
    @api.expect(style_vector_model)
    @api.marshal_with(image_model)
    def post(self):
        global stored_w_vectors
        if stored_w_vectors is None:
            return {"error": "W vectors not initialized. Call /randomize-latents first."}, 400

        # Get slider values from request
        slider_values = request.json.get("slider_values", [])

        adjusted_ws = stylegan.manipulate_w(slider_values, stored_w_vectors)

        with torch.no_grad():
            tensors, time_ = time_function(stylegan.generator.predict_from_style, adjusted_ws, stylegan.level, stylegan.alpha, False)

        images_b64 = [normalized_tensor_to_b64(tensor.unsqueeze(0)) for tensor in tensors]

        return {"images": images_b64, "time": time_}, 200

