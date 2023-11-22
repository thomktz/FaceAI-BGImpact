# FaceAI - Background Impact

This project implements various Generative AI models to generate faces:
- Deep Convolutional Generative Adversarial Network (DCGAN)
- Progressive-growing StyleGAN
- Variational Autoencoder (VAE)

The projects also implements two new versions of the [Flicker-Face-HQ (FFHQ)](https://github.com)
- ffhq_blur (Where the background is blurred)
- ffhq_grey (Where the background is greyed-out)

The motivation stemmed from the fact that a lot of the variance in VAEs seemed to be wasted on the background of the image. There are no existing large-scale faces datasets with uniform background (the ORL dataset only has 400 images), so we decided to create our own.

## Folder structure
```
faceai-bgimpact
├── configs                     # Configuration files for models
├── data_processing             # Scripts and notebooks for data preprocessing   
│   ├── paths.py                # Script to define data paths
│   ├── download_raw_ffhq.py    # Script to download raw FFHQ images
│   ├── create_masks.py         # Script to create image masks
│   ├── create_blur_and_grey.py # Script to create blurred and greyscale images
│   └── download_all_ffhq.py    # Script to download our pre-processed datasets
├── models
│   ├── dcgan_                  # DCGAN model implementation
│   │   ├── __init__.py         
│   │   ├── dcgan.py            # DCGAN model definition
│   │   ├── discriminator.py    # Discriminator part of DCGAN
│   │   └── generator.py        # Generator part of DCGAN
│   ├── stylegan_               # StyleGAN model implementation
│   │   ├── __init__.py         
│   │   ├── discriminator.py    # Discriminator part of StyleGAN
│   │   ├── generator.py        # Generator part of StyleGAN
│   │   ├── loss.py             # Loss functions for StyleGAN
│   │   ├── stylegan.py         # StyleGAN model definition
│   │   └── utils.py            # Utility functions for StyleGAN   
│   ├── abstract_model.py       # Abstract model class for common functionalities
│   ├── data_loader.py          # Data loading utilities
│   └── utils.py                # General utility functions
├── tests
│   ├── test_data_loader.py     # Pytests for data loading utilities
│   ├── test_dcgan.py           # Pytests for DCGAN model
│   └── test_stylegan.py        # Pytests for StyleGAN model
├── create_video.py             # Utility script to create videos
├── generate_images.py          # Script to generate images using models
├── graph_fids.py               # Script to graph FID scores
├── train.py                    # Script to train models
├── pyproject.toml              # Poetry package management configuration file
└── README.md


```
## Training Script (train.py)

The `train.py` script is an entry point to train a model. It includes command-line arguments to specify the model type, configuration parameters, learning rate, latent dimension, batch size, number of epochs, and intervals for saving.

## How to Use

0. Ensure you have the required dependencies installed (see Dependencies section below).

1. To train a model, you need to specify the model type using the `--model` flag, and the dataset with the `--dataset` flag. 

2. Additional command-line arguments allow for fine-tuning the training process:

Model arguments:
- `--model`: Required. Specifies the type of model to train with options "DCGAN", "StyleGAN".
- `--dataset`: Required. Selects the dataset to use with options "ffhq_raw", "ffhq_blur", "ffhq_grey".
- `--latent-dim`: Optional. Defines the dimension of the latent space for generative models.
- `--config-path`: Optional. Path to a custom JSON configuration file for model training.

Training arguments:
- `--lr`: Optional. Learning rate for DCGAN model training.
- `--dlr`: Optional. Discriminator learning rate for StyleGAN training.
- `--glr`: Optional. Generator learning rate for StyleGAN training.
- `--mlr`: Optional. W-Mapping learning rate for StyleGAN training.
- `--loss`: Optional. Specifies the loss function to use with defaults to "wgan-gp"; choices are "wgan", "wgan-gp", "basic".
- `--batch-size`: Optional. Defines the batch size during training.
- `--num-epochs`: Optional. Sets the number of epochs for training the model.
- `--save-interval`: Optional. Epoch interval to wait before saving models and generated images.
- `--image-interval`: Optional. Iteration interval to wait before saving generated images.

Checkpoint arguments:
- `--list`: Optional. Lists all available checkpoints if set.
- `--checkpoint-path`: Optional. Path to a specific checkpoint file to resume training, takes precedence over `--checkpoint-epoch`.
- `--checkpoint-epoch`: Optional. Specifies the epoch number from which to resume training.


3. Example usage:

`python train.py --model StyleGAN --dataset ffhq_raw`

You can also provide additional arguments as needed.

## Dependencies

We use [Poetry](https://python-poetry.org/) for dependency management. To install the dependencies, you should use the provided poetry environment. If you do not have poetry installed, you can install it using pip:

`pip install poetry`

Then, you can install the dependencies using:

`poetry install`

And activate the environment using:

`poetry shell`

Since these packages are heavy (especially PyTorch), you may use your own environment if you wish, but it might not work as expected.

Training on GPU is highly recommended. If you have a CUDA-enabled GPU, you should install the CUDA version of PyTorch.

## Testing

Automated tests can be run using PyTest. Ensure you have PyTest installed and run:

`pytest -v`

in the project root to execute all tests.

Alternatively, with poetry, you can run:

`poetry run pytest -v`

## TODO:
- VAE
- BMSG-GAN (https://github.com/akanimax/BMSG-GAN)