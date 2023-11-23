config = {
    "dataset": "ffhq_raw",
    "dlr": 0.0005,
    "glr": 0.001,
    "mlr": 0.00001,
    "loss": "wgan-gp",
    "latent_dim": 256,
    "w_dim": 256,
    "style_layers": 8,
    "batch_size": 16,
    "save_interval": 5, 
    "image_interval": 100,
    "level_epochs": {
        0: {
            "transition": 0,
            "training": 4
        },           
        1: {
            "transition": 3,
            "training": 5
        },
        2: {
            "transition": 5,
            "training": 5
        },
        3: {
            "transition": 10,
            "training": 10
        },
        4: {
            "transition": 20,
            "training": 20
        },
        5: {
            "transition": 20,
            "training": 50
        }
    }
}