from faceai_bgimpact.models import StyleGAN, VAE


class ModelManager:
    """Manage different StyleGAN models at once."""

    def __init__(self):
        self.models = {}

    def load_all_models(self, model_type, datasets, num_pca_samples=200000, num_pca_components=100):
        """
        Load all models and fit PCA on them.

        Parameters:
        ----------
        model_types : list of str
            The types of the models to load.
        datasets : list of str
            The names of the datasets to load.
        num_pca_samples : int
            The number of samples to use for PCA.
        num_pca_components : int
            The number of components to use for PCA.
        """
        for model_name in datasets:
            dataset_name = "ffhq_" + model_name
            if model_type == "StyleGAN":
                self.models[model_name] = StyleGAN.from_checkpoint(
                    dataset_name=dataset_name,
                    checkpoint_path=f"models/StyleGAN_{dataset_name}.pth",
                    loss="r1",
                    device="cpu",
                )
            elif model_type == "VAE":
                self.models[model_name] = VAE.from_checkpoint(
                    checkpoint_path=f"models/VAE_{dataset_name}.pth",
                    device="cpu",
                )
            else:
                raise ValueError(f"Unknown model type {model_type}.")
            self.models[model_name].fit_pca(
                num_samples=num_pca_samples, batch_size=100, n_components=num_pca_components
            )
            print(f"Model {model_type} {dataset_name} loaded and PCA fitted.")

    def __getitem__(self, key):
        """Magic method to get a model by name."""
        return self.models[key]


if __name__ == "__main__":
    # For generating the PCA components inside a __main__ block
    # This is needed to prevent a crash from the multiprocessing library

    model_manager = ModelManager()
    model_manager.load_all_models("StyleGAN", ["grey", "raw"])
    model_manager.load_all_models("VAE", ["grey", "raw"], num_pca_samples=69000)
    print("Models loaded.")
