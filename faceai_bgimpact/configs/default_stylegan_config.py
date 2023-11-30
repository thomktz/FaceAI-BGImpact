config = {
    "dataset": "ffhq_raw",
    "dlr": 0.0006,
    "glr": 0.0008,
    "mlr": 0.000008,
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
            "training": 10
        },           
        1: {
            "transition": 3,
            "training": 20
        },
        2: {
            "transition": 10,
            "training": 75
        },
        3: {
            "transition": 20,
            "training": 85
        },
        4: {
            "transition": 25,
            "training": 90
        },
        5: {
            "transition": 30,
            "training": 100
        }
    }
}