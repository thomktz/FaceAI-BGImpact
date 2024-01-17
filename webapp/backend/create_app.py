import os

from flask import Flask
from flask_cors import CORS

from werkzeug.middleware.proxy_fix import ProxyFix
from backend.extensions import api
from backend.api import stylegan_ns


def create_app():
    """Create an app instance."""
    print(f"Creating app from {os.getcwd()}", flush=True)
    print("API KEY:", os.environ.get("API_KEY"))

    # Initialize the Flask app
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or b"\x05\xe1C\x07k\x1ay<\xb6\xa4\xf8\xc6\xa8f\xb4*"
    app.config.update(
        DEBUG=True,
        SESSION_COOKIE_HTTPONLY=True,
        REMEMBER_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
    )
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)

    CORS(
        app,
        supports_credentials=True,
        resources={
            r"/*": {
                "origins": [
                    "http://localhost:8080",
                    "http://127.0.0.1:8080",
                    "https://aml.kientz.net",
                    "http://89.168.39.28:8080" "https://89.168.39.28:8080",
                ],
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept"],
                "expose_headers": ["Access-Control-Allow-Origin", "Access-Control-Allow-Credentials"],
            }
        },
    )

    # Initialize the extensions
    api.init_app(app)

    # Initialize the Flask-RestX Api and register the namespaces
    api.add_namespace(stylegan_ns, path="/stylegan")
    # api.add_namespace(vae_ns, path="/vae")

    # For testing purposes
    @app.route("/hello")
    def index():
        return "Hello, World!"

    return app
