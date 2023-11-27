from flask_restx import fields
from .namespace import api

image_model = api.model("Image", {
    "images": fields.List(fields.String, description="A base64 encoded string of the generated image."),
    "time": fields.Float(description="The time it took to generate the image."),
})

latent_vector_model = api.model("LatentVector", {
    "latent_vector": fields.List(fields.Float, required=True, description="The latent vector to generate the image from.", min_items=1)
})

style_vector_model = api.model("StyleVector", {
    "style_vector": fields.List(fields.Float, required=True, description="The style vector to generate the image from.", min_items=1)
})