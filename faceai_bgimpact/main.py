# main.py within the faceai_bgimpact package

import argparse
from faceai_bgimpact.scripts.train import train_function
from faceai_bgimpact.scripts.create_video import create_video_function

def main():
    parser = argparse.ArgumentParser(description="FaceAI-BGImpact command-line tool")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Subparser for the 'train' command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument("--model", type=str, required=True, choices=["DCGAN", "StyleGAN"], help="Type of model to train")
    train_parser.add_argument("--dataset", type=str, required=True, choices=["ffhq_raw", "ffhq_blur", "ffhq_grey"], help="Name of the dataset to use")
    train_parser.add_argument("--latent-dim", type=int, help="Dimension of the latent space")
    train_parser.add_argument("--config-path", type=str, default=None, help="Path to a custom JSON configuration file")
    train_parser.add_argument("--lr", type=float, help="Learning rate (for DCGAN)")
    train_parser.add_argument("--dlr", type=float, help="Discriminator learning rate (for StyleGAN)")
    train_parser.add_argument("--glr", type=float, help="Generator learning rate (for StyleGAN)")
    train_parser.add_argument("--mlr", type=float, help="W-Mapping learning rate (for StyleGAN)")
    train_parser.add_argument("--loss", type=str, choices=["wgan", "wgan-gp", "basic"], default="wgan-gp", help="Learning rate decay")
    train_parser.add_argument("--batch-size", type=int, help="Batch size")
    train_parser.add_argument("--num-epochs", type=int, help="Number of epochs to train (DCGAN)")
    train_parser.add_argument("--save-interval", type=int, help="Number of epochs to wait before saving models and images")
    train_parser.add_argument("--image-interval", type=int, help="Number of iterations to wait before saving generated images")
    train_parser.add_argument("--list", action="store_true", help="List available checkpoints")
    train_parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to a checkpoint file to resume training. Has priority over --checkpoint-epoch")
    train_parser.add_argument("--checkpoint-epoch", type=int, default=None, help="Epoch number of the checkpoint to resume training from")

    # Subparser for the 'create-video' command
    create_video_parser = subparsers.add_parser('create-video', help='Create a video from training frames')
    create_video_parser.add_argument("--model", type=str, required=True, choices=["DCGAN", "StyleGAN"], help="Type of model")
    create_video_parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset used")
    create_video_parser.add_argument("--frame-rate", type=int, default=30, help="Frame rate for the video")

    args = parser.parse_args()

    if args.command == 'train':
        # Call the train function with all required arguments
        print(f"\n##############" + "#"*len(f" {args.model} - {args.dataset} ") +"##################")
        print(f"################ {args.model} - {args.dataset} ################")
        print(f"##############" + "#"*len(f" {args.model} - {args.dataset} ") +"##################\n")
        train_function(
            model=args.model,
            dataset=args.dataset,
            latent_dim=args.latent_dim,
            config_path=args.config_path,
            lr=args.lr,
            dlr=args.dlr,
            glr=args.glr,
            mlr=args.mlr,
            loss=args.loss,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            save_interval=args.save_interval,
            image_interval=args.image_interval,
            list_checkpoints_flag=args.list,
            checkpoint_path=args.checkpoint_path,
            checkpoint_epoch=args.checkpoint_epoch
        )
    elif args.command == 'create-video':
        print(f"\n##############" + "#"*len(f" {args.model} - {args.dataset} ") +"##################")
        print(f"################ {args.model} - {args.dataset} ################")
        print(f"##############" + "#"*len(f" {args.model} - {args.dataset} ") +"##################\n")
        create_video_function(
            model=args.model,
            dataset=args.dataset,
            frame_rate=args.frame_rate
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
