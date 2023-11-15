import pytest
from models.data_loader import get_dataloaders
from torchvision import transforms

@pytest.mark.parametrize("dataset_name", ["ffhq_raw", "ffhq_blur", "ffhq_grey"])
def test_dataset_loading(dataset_name):
    train_loader, test_loader = get_dataloaders(dataset_name, batch_size=10)
    
    # Check if train and test loaders are not empty
    assert len(train_loader) > 0, f"Train loader for {dataset_name} is empty"
    assert len(test_loader) > 0, f"Test loader for {dataset_name} is empty"

    # Check a batch from each loader
    for loader in [train_loader, test_loader]:
        images = next(iter(loader))
        assert images.shape == (10, 3, 128, 128), f"Image batch size or dimensions are incorrect for {dataset_name}"
