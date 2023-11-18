# AML-VAE-GAN
Advanced Machine Learning project: Effect of removing the background from training images on Variational AutoEncoders and Generative Adversarial Networks.

## Motivation
- ORL dataset only has 400 images.
- A lot of the latent space of VAEs and GANs seems to be captured by the image background in generation tasks, instead of the subject.

## Folder structure
```
├── configs                   # Configuration files for models
│ └── default_gan_config.json # Default GAN configuration
├── data_processing           # Scripts for data preprocessing and organization
│ ├── create_blur_and_grey.py # Script to create blurred and grey variants of images
│ ├── create_masks.py         # Script to create masks for images
│ ├── download_raw_ffhq.py    # Script to download the raw FFHQ dataset
│ ├── download_all_ffhq.py    # Script to download all FFHQs dataset
│ └── paths.py                # Utility script to define path constants
├── models                    # Model definitions and utilities
│ ├── data_loader.py          # Data loading utilities for GAN training
│ ├── abstract_model.py       # Abstract base class for our models
│ └── dcgan.py                  # DCGAN model definition
├── tests                     # Automated tests for the project
│ ├── test_data_loader.py     # Tests for the data loader utility
│ └── test_dcgan.py             # Tests for the DCGAN model
└── train.py             # Main training script for the models
```
## Training Script (train.py)

The `train.py` script is an entry point to train a model. It includes command-line arguments to specify the model type, configuration parameters, learning rate, latent dimension, batch size, number of epochs, and intervals for saving.

## How to Use

1. To train a model, you need to specify the model type using the `--model-type` flag, and optionally provide a path to a custom JSON configuration file with `--config-path`.

2. Additional command-line arguments allow for fine-tuning the training process:

- `--model-type`: Required. The type of model to train (options: "DCGAN", "VAE").
- `--dataset-name`: Required. The dataset to use (options: "ffhq_raw", "ffhq_blur", "ffhq_grey").
- `--config-path`: Path to a custom JSON configuration file.
- `--lr`: Learning rate for the model training.
- `--latent-dim`: Dimension of the model's latent space.
- `--batch-size`: Number of samples per batch during training.
- `--num-epochs`: Total number of epochs for which to train the model.
- `--log-interval`: Batches to process between logging training progress.
- `--save-interval`: Epochs to complete before saving models and images.
- `--checkpoint-path`: Path to a checkpoint file to resume training (overrides `--checkpoint-epoch`).
- `--checkpoint-epoch`: Epoch number from which to resume training.

3. To start training, run:

`python train.py --model-type gan`

You can also provide additional arguments as needed.

## Testing

Automated tests can be run using PyTest. Ensure you have PyTest installed and run:

`pytest -s`

in the project root to execute all tests.

## Dependencies

This project requires the following main dependencies:

- PyTorch
- torchvision
- tqdm
- pytest (for running tests)

To install the dependencies, you should use the provided poetry environment. If you do not have poetry installed, you can install it using pip:

`pip install poetry`

Then, you can install the dependencies using:

`poetry install`

And activate the environment using:

`poetry shell`
