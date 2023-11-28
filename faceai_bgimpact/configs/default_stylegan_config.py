config = {
    "dataset": "ffhq_raw",
    "dlr": 0.0005,
    "glr": 0.0007,
    "mlr": 0.000007,
    "loss": "wgan-gp",
    "latent_dim": 256,
    "w_dim": 256,
    "style_layers": 8,
    "batch_size": 256,
    "save_interval": 1, 
    "image_interval": 20,
    "level_epochs": {
        0: {
            "transition": 0,
            "training": 5
        },           
        1: {
            "transition": 3,
            "training": 7
        },
        2: {
            "transition": 10,
            "training": 15
        },
        3: {
            "transition": 15,
            "training": 30
        },
        4: {
            "transition": 25,
            "training": 30
        },
        5: {
            "transition": 30,
            "training": 50
        }
    }
}