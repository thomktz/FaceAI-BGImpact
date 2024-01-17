# FaceAI - Background Impact

This project implements various Generative AI models to generate faces:

- Deep Convolutional Generative Adversarial Network (DCGAN)
- Progressive-growing StyleGAN
- Variational Autoencoder (VAE)

The projects also implements two new versions of the [Flicker-Face-HQ (FFHQ)](https://github.com)

- [FFHQ-Blur](https://www.kaggle.com/datasets/thomaskientz/ffhq-blur) (Where the background is blurred)
- [FFHQ-Grey](https://www.kaggle.com/datasets/thomaskientz/ffhq-grey) (Where the background is greyed-out)

![387467294_852111396644872_6368954973460925603_n](https://github.com/thomktz/FaceAI-BGImpact/assets/60552083/d2a015eb-eabe-4a9c-ad6e-7ed35051241f)

The motivation stemmed from the fact that a lot of the variance in VAEs seemed to be wasted on the background of the image. There are no existing large-scale faces datasets with uniform background (the ORL dataset only has 400 images), so we decided to create our own.

## Installation

### Pip

We published the models and dataset transformations as a pip package. To install, run:

`pip install faceai-bgimpact`

## Folder structure

```
FaceAI-BGImpact
├── faceai_bgimpact
│   ├── configs                         # Configuration files for models
│   │   ├── default_dcgan_config.py
│   │   └── default_stylegan_config.py
│   │   └── default_vae_config.py
│   ├── data_processing                 # Scripts and notebooks for data preprocessing
│   │   ├── paths.py                    # Data paths
│   │   ├── download_raw_ffhq.py        # Functions to download raw FFHQ dataset
│   │   ├── create_masks.py             # Functions to create masks for FFHQ dataset
│   │   ├── create_blur_and_grey.py     # Functions to create blurred and greyed-out FFHQ datasets
│   │   └── download_all_ffhq.py        # Functions to download all FFHQ datasets
│   ├── models
│   │   ├── provae                      # PROVAE model implementation
│   │   │   ├── provae.py
│   │   ├── dcgan_                      # DCGAN model implementation
│   │   │   ├── dcgan.py
│   │   │   ├── discriminator.py
│   │   │   └── generator.py
│   │   ├── stylegan_                   # StyleGAN model implementation
│   │   │   ├── discriminator.py
│   │   │   ├── generator.py
│   │   │   ├── loss.py                 # Loss functions for StyleGAN
│   │   │   └── stylegan.py
|   |   ├── vae_                        # VAE model implementation
|   |   |   ├── decoder.py              
│   │   │   ├── encoder.py              
│   │   │   └── vae.py                  
│   │   ├── abstract_model.py           # Abstract model class for common functionalities
│   │   ├── data_loader.py              # Data loading utilities
│   │   └── utils.py
│   ├── scripts
│   │   ├── train.py                    # Script to train models
│   │   ├── create_video.py             # Script to create videos from generated images
│   │   ├── generate_images.py
│   │   └── graph_fids.py
│   ├── main.py                         # Entry point for the package
│   └── Nirmala.ttf                     # Font file used in the project
├── tests                               # Pytests
├── README.md
└── pyproject.toml                      # Poetry package management configuration file
```

## Training Script (train.py)

The `train.py` script is an entry point to train a model. It includes command-line arguments to specify the model type, configuration parameters, learning rate, latent dimension, batch size, number of epochs, and intervals for saving.

## How to Use

0. Ensure you have the package installed, or the required dependencies for dev installed (see Dependencies section below).

1. To train a model, you call `faceai-bgimpact train` and specify the model type using the `--model` flag, and the dataset with the `--dataset` flag.

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

`faceai-bgimpact download-all-ffhq`

`faceai-bgimpact train --model StyleGAN --dataset ffhq_raw`

`faceai-bgimpact create-video --model StyleGAN --dataset ffhq_raw`

## Dependencies

We use [Poetry](https://python-poetry.org/) for dependency management. To install the dependencies, you should use the provided poetry environment. If you do not have poetry installed, you can install it using pip:

`pip install poetry`

Then, you can install the dependencies using:

`poetry install`

And activate the environment using:

`poetry shell`

To install the package locally, you can run

`poetry install`

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
- GANSpace (https://proceedings.neurips.cc/paper/2020/file/6fe43269967adbb64ec6149852b5cc3e-Paper.pdf)
