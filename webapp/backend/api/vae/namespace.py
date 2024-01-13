from flask_restx import Namespace

api = Namespace("vae", description="Operations related to VAE")

from . import routes
