from faceai_bgimpact.models import StyleGAN
from faceai_bgimpact.scripts.train import find_checkpoint_path


def graph_fids_function(model, dataset, checkpoint_epoch):
    """
    Graph the FID scores for a model.

    Parameters:
    ----------
    model : str
        The model type to use.
    dataset : str
        The dataset to use.
    checkpoint_epoch : int
        The epoch of the checkpoint to use.
    """
    # Get the checkpoint path
    checkpoint_dir = f"outputs/{model}_checkpoints_{dataset}"
    checkpoint_path = find_checkpoint_path(checkpoint_dir, checkpoint_epoch)

    # Load the model
    model = StyleGAN.from_checkpoint(
        dataset_name=dataset,
        checkpoint_path=checkpoint_path,
        loss="wgan-gp",
        device="cpu",
    )

    # Plot the FID scores
    model.graph_fid()
