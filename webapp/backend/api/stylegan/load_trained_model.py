from faceai_bgimpact.models import StyleGAN

stylegan = StyleGAN.from_checkpoint(
    dataset_name="ffhq_raw",
    checkpoint_path="models/StyleGAN.pth",
    loss="wgan-gp",
    device="cpu",
)