import base64
import torch
from io import BytesIO
from flask_restx import Resource
from torchvision.transforms import ToPILImage

from .namespace import api
from .model_manager import ModelManager
from flask_restx import fields


model_manager = ModelManager()
model_manager.load_all_models(["grey", "raw"])  # blur

stored_w_vectors = None
APPLY_NOISE = False

to_pil_image = ToPILImage()

x1 = None
base_w = None
x1_others = None
n_sliders = None
latent_dim = 256


@api.route("/set-n-sliders")
class SetNSliders(Resource):
    @api.expect(api.model("NSliders", {"n_sliders": fields.Integer(required=True)}))
    def post(self):
        global n_sliders, x1_others
        n_sliders = api.payload["n_sliders"]
        model = model_manager[api.payload["model_name"].lower()]
        N_PCA = model.pca.n_components_
        x1_others = torch.zeros(N_PCA - n_sliders)
        return {"message": f"n_sliders set to {n_sliders}"}, 200


@api.route("/random-x1-others")
class RandomX1Others(Resource):
    @api.expect(api.model("RandomX1Others", {"model_name": fields.String(required=True)}))
    def post(self):
        global x1_others
        model = model_manager[api.payload["model_name"].lower()]
        N_PCA = model.pca.n_components_
        x1_others = torch.randn(N_PCA - n_sliders) / 5
        return {"message": "Random x1 others set"}, 200


@api.route("/zero-x1-others")
class ZeroX1Others(Resource):
    @api.expect(api.model("ZeroX1Others", {"model_name": fields.String(required=True)}))
    def post(self):
        global x1_others
        model = model_manager[api.payload["model_name"].lower()]
        N_PCA = model.pca.n_components_
        x1_others = torch.zeros(N_PCA - n_sliders)
        return {"message": "Zero x1 others set"}, 200


@api.route("/x1-sliders", methods=["POST"])
class X1Sliders(Resource):
    @api.expect(
        api.model("X1Sliders", {"slider_values": fields.List(fields.Float), "model_name": fields.String(required=True)})
    )
    def post(self):
        global x1_others, x1, base_w
        model = model_manager[api.payload["model_name"].lower()]
        x1 = torch.cat([torch.tensor(api.payload["slider_values"]), x1_others])
        base_w = torch.tensor(model.pca.inverse_transform(x1))
        return {"message": "Main slider values updated"}, 200


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
                "layers_list": fields.List(
                    fields.List(fields.Integer),
                    required=True,
                    description="Layer indices for each eigenvector",
                ),
                "model_name": fields.String(required=True, description="Model name"),
            },
        )
    )
    def post(self):
        eigenvector_strengths = api.payload["eigenvector_strengths"]
        layers_list = api.payload["layers_list"]
        model = model_manager[api.payload["model_name"].lower()]

        # Apply blend_styles to generate the blended latent vectors
        image_tensor = model.blend_styles(base_w, x1, eigenvector_strengths, layers_list, APPLY_NOISE)
        image_tensor = (image_tensor.clamp(-1, 1) + 1) / 2  # Denormalize the image

        # Convert tensor to Base64 image
        pil_image = to_pil_image(image_tensor.squeeze(0))
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {"image": img_base64}, 200


# @api.route("/generate-transition-images")
# class GenerateTransitionImages(Resource):
#     @api.expect(
#         api.model(
#             "GenerateTransitionImages",
#             {
#                 "max_eigenvector_index": fields.Integer(
#                     required=True,
#                     description="Maximum index of the eigenvector to consider",
#                 ),
#                 "num_steps": fields.Integer(
#                     required=True,
#                     default=10,
#                     description="Number of steps in the transition",
#                 ),
#                 "min_strength": fields.Float(
#                     required=True,
#                     default=-2.0,
#                     description="Minimum strength of eigenvector",
#                 ),
#                 "max_strength": fields.Float(
#                     required=True,
#                     default=2.0,
#                     description="Maximum strength of eigenvector",
#                 ),
#             },
#         )
#     )
#     def post(self):
#         max_eigenvector_index = api.payload["max_eigenvector_index"]
#         num_steps = api.payload["num_steps"]
#         min_strength = api.payload["min_strength"]
#         max_strength = api.payload["max_strength"]

#         # Create directory for images if it doesn't exist
#         if not os.path.exists("eig_effects"):
#             os.makedirs("eig_effects")

#         num_layers = 5

#         # Initialize base_w and x1
#         base_w_ = torch.tensor(model_manager.current_model.pca.inverse_transform(torch.randn(N_PCA) / 5))
#         x1_ = torch.zeros(N_PCA)

#         for eigenvector_index in range(max_eigenvector_index + 1):
#             for layer in range(num_layers):
#                 concatenated_images = []
#                 eigenvector_strengths = [0] * (eigenvector_index + 1)
#                 layers_list = [list(range(5))] * (eigenvector_index + 1)
#                 layers_list[eigenvector_index] = [layer]

#                 for step in range(num_steps):
#                     # Update eigenvector strength
#                     strength = min_strength + (max_strength - min_strength) * step / (num_steps - 1)
#                     eigenvector_strengths[eigenvector_index] = strength

#                     # Generate blended latent vector
#                     image_tensor = model_manager.current_model.blend_styles(
#                         base_w_, x1_, eigenvector_strengths, layers_list, APPLY_NOISE
#                     )
#                     image_tensor = (image_tensor.clamp(-1, 1) + 1) / 2  # Denormalize the image

#                     # Convert tensor to PIL image and add to the list
#                     pil_image = to_pil_image(image_tensor.squeeze(0))
#                     # Resize image to 200x200
#                     resized_image = pil_image.resize((200, 200))
#                     concatenated_images.append(resized_image)

#                 # Concatenate images horizontally
#                 concat_image = np.concatenate([np.array(img) for img in concatenated_images], axis=1)
#                 concat_image = Image.fromarray(concat_image)

#                 # Save concatenated image
#                 concat_image_path = f"eig_effects/eig_{eigenvector_index}_{layer}.png"
#                 concat_image.save(concat_image_path)
