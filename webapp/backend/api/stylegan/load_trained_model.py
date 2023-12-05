from faceai_bgimpact.models import StyleGAN

stylegan = StyleGAN.from_checkpoint(
    dataset_name="ffhq_grey",
    checkpoint_path="models/StyleGAN.pth",
    loss="r1",
    device="cpu",
)

print("Fitting PCA...")
stylegan.fit_pca(num_samples=70000, batch_size=100, n_components=200)
print("Done!")
