import pytest
from models.data_loader import get_dataloader

@pytest.mark.parametrize("dataset_name", ["ffhq_raw", "ffhq_blur", "ffhq_grey"])
def test_dataset_loading(dataset_name):
    resolution = 32
    alpha = 0.5
    batch_size = 10
    dataset, loader = get_dataloader(dataset_name, batch_size=batch_size, resolution=resolution, alpha=alpha)
    
    # Check if train and test loaders are not empty
    assert len(dataset) > 0, f"Dataset for {dataset_name} is empty"
    assert len(loader) > 0, f"Loader for {dataset_name} is empty"

    images = next(iter(loader))
    assert images.shape == (batch_size, 3, resolution, resolution), f"Image batch size or dimensions are incorrect for {dataset_name}"
