import base64
import torch
from io import BytesIO
from flask_restx import Resource
from torchvision.transforms import ToPILImage

from .namespace import api
from ..model_manager import ModelManager
from flask_restx import fields

to_pil_image = ToPILImage()

model_manager = ModelManager()
model_manager.load_all_models("VAE", ["grey", "raw"], num_pca_samples=69000)

x_others = torch.zeros(512)
latent_dim = 128


@api.route("/set-n-sliders")
class SetNSliders(Resource):
    @api.expect(api.model("NSliders", {"n_sliders": fields.Integer(required=True)}))
    def post(self):
        global n_sliders, x_others
        n_sliders = api.payload["n_sliders"]
        model = model_manager[api.payload["model_name"].lower()]
        N_PCA = model.pca.n_components_
        x_others = torch.zeros(N_PCA - n_sliders)
        return {"message": f"n_sliders set to {n_sliders}"}, 200


@api.route("/random-x-others")
class RandomxOthers(Resource):
    @api.expect(api.model("RandomxOthers", {"model_name": fields.String(required=True)}))
    def post(self):
        global x_others
        model = model_manager[api.payload["model_name"].lower()]
        N_PCA = model.pca.n_components_
        x_others = torch.randn(N_PCA - n_sliders) / 5
        return {"message": "Random x others set"}, 200


@api.route("/zero-x-others")
class ZeroxOthers(Resource):
    @api.expect(api.model("ZeroxOthers", {"model_name": fields.String(required=True)}))
    def post(self):
        global x_others
        model = model_manager[api.payload["model_name"].lower()]
        N_PCA = model.pca.n_components_
        x_others = torch.zeros(N_PCA - n_sliders)
        return {"message": "Zero x others set"}, 200


@api.route("/generate-image")
class GenerateImage(Resource):
    @api.expect(
        api.model(
            "GenerateImage",
            {
                "eigenvector_strengths": fields.List(
                    fields.Float,
                    required=True,
                    description="Strengths for each eigenvector",
                ),
                "model_name": fields.String(required=True, description="Model name"),
            },
        )
    )
    def post(self):
        eigenvector_strengths = api.payload["eigenvector_strengths"]
        model = model_manager[api.payload["model_name"].lower()]

        # Apply blend_styles to generate the blended latent vectors
        image_tensor = model.image_from_eigenvector_strengths(eigenvector_strengths)

        # Convert tensor to Base64 image
        pil_image = to_pil_image(image_tensor.squeeze(0))
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {"image": img_base64}, 200
