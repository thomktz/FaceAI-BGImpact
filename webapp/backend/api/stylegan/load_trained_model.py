from faceai_bgimpact.models import StyleGAN

stylegan = StyleGAN.from_checkpoint(
    dataset_name="ffhq_raw",
    checkpoint_path="models/StyleGAN.pth",
    loss="wgan-gp",
    device="cpu",
)

print("Fitting PCA...")
stylegan.fit_pca(num_samples=1000, batch_size=100, n_components=50)
print("Done!")
