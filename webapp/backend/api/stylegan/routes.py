import base64
import torch
from io import BytesIO
from time import time
from flask import request
from flask_restx import Resource
from torchvision.transforms import ToPILImage

from .namespace import api
from .load_trained_model import stylegan
from .models import image_model, latent_vector_model, style_vector_model
from faceai_bgimpact.models.data_loader import denormalize_image

to_pil_image = ToPILImage()

def time_function(func, *args, **kwargs):
    start_time = time()
    result = func(*args, **kwargs)
    end_time = time()
    elapsed_time = end_time - start_time
    return result, elapsed_time

def normalized_tensor_to_b64(tensor):
    # Generate image and denormalize
    dn_tensor = denormalize_image(tensor)
    
    # Convert to PIL Image
    image = to_pil_image(dn_tensor.squeeze(0))
    
    # Convert the PIL Image to a BytesIO object
    buffered = BytesIO()
    image.save(buffered, format="JPEG")

    # Encode the image as a base64 string
    return base64.b64encode(buffered.getvalue()).decode()
    

@api.route("/random")
class GenerateRandomImage(Resource):
    @api.marshal_with(image_model)
    def get(self):
        # Get latent space dimensions
        latent_dim = stylegan.latent_dim
        
        # Generate random tensor, send to CPU
        z = torch.randn(1, latent_dim).to("cpu")

        # Encode the image as a base64 string
        with torch.no_grad():
            tensor, time = time_function(stylegan.generator, z, stylegan.level, stylegan.alpha)
            
        img_str = normalized_tensor_to_b64(tensor)

        # Return the base64 string
        return {"image": img_str, "time": time}, 200
    
@api.route("/from-latent")
class GenerateFromLatent(Resource):
    @api.expect(latent_vector_model)
    @api.marshal_with(image_model)
    def post(self):
        # Get the latent vector from the request"s JSON
        input_vector = request.json.get("latent_vector", [])
        
        # Pad the latent vector with zeros if it's shorter than latent_dim
        latent_dim = stylegan.latent_dim
        padded_vector = input_vector + [0] * (latent_dim - len(input_vector))
        
        # Ensure the vector is not longer than latent_dim
        latent_vector = torch.tensor(padded_vector[:latent_dim], dtype=torch.float32).unsqueeze(0).to("cpu")
        
        # Generate image and denormalize
        with torch.no_grad():  # Ensure no gradients are calculated
            tensor, time = time_function(stylegan.generator, latent_vector, stylegan.level, stylegan.alpha)
            
        img_str = normalized_tensor_to_b64(tensor)
        
        return {"image": img_str, "time": time}, 200
    

@api.route("/from-style")
class GenerateFromStyle(Resource):
    @api.expect(style_vector_model)
    @api.marshal_with(image_model)
    def post(self):
        # Get the latent vector from the request"s JSON
        input_vector = request.json.get("latent_vector", [])
        
        # Pad the latent vector with zeros if it's shorter than w_dim
        w_dim = stylegan.w_dim
        padded_vector = input_vector + [0] * (w_dim - len(input_vector))
        
        # Ensure the vector is not longer than w_dim
        latent_vector = torch.tensor(padded_vector[:w_dim], dtype=torch.float32).unsqueeze(0).unsqueeze(2).unsqueeze(3).to("cpu")
        
        # Generate image and denormalize
        with torch.no_grad():  # Ensure no gradients are calculated
            tensor, time = time_function(stylegan.generator.predict_from_style, latent_vector, stylegan.level, stylegan.alpha, False)
            
        img_str = normalized_tensor_to_b64(tensor)
        
        return {"image": img_str, "time": time}, 200