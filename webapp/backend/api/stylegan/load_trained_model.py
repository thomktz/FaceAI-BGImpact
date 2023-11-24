from faceai_bgimpact.models import StyleGAN

stylegan = StyleGAN.from_checkpoint(
    "ffhq_raw",
    "models/StyleGAN.pth",
    "loss"
    "cpu"
)