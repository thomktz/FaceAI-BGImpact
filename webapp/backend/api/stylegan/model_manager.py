from faceai_bgimpact.models import StyleGAN


class ModelManager:
    """Manage different StyleGAN models at once."""

    def __init__(self):
        self.models = {}

    def load_all_models(self, model_names, num_pca_samples=200000, num_pca_components=100):
        """
        Load all models and fit PCA on them.

        Parameters:
        ----------
        model_names : list of str
            The names of the models to load.
        num_pca_samples : int
            The number of samples to use for PCA.
        num_pca_components : int
            The number of components to use for PCA.
        """
        for model_name in model_names:
            dataset_name = "ffhq_" + model_name
            self.models[model_name] = StyleGAN.from_checkpoint(
                dataset_name=dataset_name,
                checkpoint_path=f"models/StyleGAN_{dataset_name}.pth",
                loss="r1",
                device="cpu",
            )
            self.models[model_name].fit_pca(
                num_samples=num_pca_samples, batch_size=100, n_components=num_pca_components
            )
            print(f"Model for {dataset_name} loaded and PCA fitted.")

    def __getitem__(self, key):
        """Magic method to get a model by name."""
        return self.models[key]
