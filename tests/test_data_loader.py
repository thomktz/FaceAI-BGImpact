import pytest
from models.data_loader import get_dataloader

@pytest.mark.parametrize("dataset_name", ["ffhq_raw", "ffhq_blur", "ffhq_grey"])
def test_dataset_loading(dataset_name):
    loader = get_dataloader(dataset_name, batch_size=10)
    
    # Check if train and test loaders are not empty
    assert len(loader) > 0, f"Loader for {dataset_name} is empty"

    images = next(iter(loader))
    assert images.shape == (10, 3, 128, 128), f"Image batch size or dimensions are incorrect for {dataset_name}"
