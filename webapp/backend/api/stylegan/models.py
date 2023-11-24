from flask_restx import Namespace

api = Namespace('stylegan', description='Operations related to StyleGAN')

from . import routes