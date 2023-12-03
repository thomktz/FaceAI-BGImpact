config = {
    "dataset": "ffhq_raw",
    "dlr": 0.0005,
    "glr": 0.0007,
    "mlr": 0.000007,
    "loss": "wgan-gp",
    "latent_dim": 256,
    "w_dim": 256,
    "style_layers": 6,
    "batch_size": 256,
    "save_interval": 1, 
    "image_interval": 40,
    "level_epochs": {
        0: {
            "transition": 0,
            "training": 15,
            "batch_size": 512
        },           
        1: {
            "transition": 4,
            "training": 20,
            "batch_size": 512
        },
        2: {
            "transition": 10,
            "training": 75,
            "batch_size": 256
        },
        3: {
            "transition": 20,
            "training": 85,
            "batch_size": 256
        },
        4: {
            "transition": 25,
            "training": 120,
            "batch_size": 128
        },
        5: {
            "transition": 40,
            "training": 100,
            "batch_size": 64
        }
    }
}