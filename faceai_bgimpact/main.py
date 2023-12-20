import argparse
from faceai_bgimpact.scripts.train import train_function
from faceai_bgimpact.scripts.create_video import create_video_function
from faceai_bgimpact.scripts.download_all_ffhq import download_all_ffhq
from faceai_bgimpact.scripts.graph_fids import graph_fids_function
from faceai_bgimpact.data_processing.create_masks import create_masks
from faceai_bgimpact.scripts.create_blur_and_grey import create_blur_and_grey


def main():
    """Parse command-line arguments and call the appropriate function."""
    parser = argparse.ArgumentParser(description="FaceAI-BGImpact command-line tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subparser for the 'train' command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["DCGAN", "StyleGAN", "VAE"],
        help="Type of model to train",
    )
    train_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ffhq_raw", "ffhq_blur", "ffhq_grey"],
        help="Name of the dataset to use",
    )
    train_parser.add_argument("--batch-size", type=int, help="Batch size")
    train_parser.add_argument("--latent-dim", type=int, help="Dimension of the latent space")
    train_parser.add_argument("--lr", type=float, help="Learning rate (for DCGAN)")
    train_parser.add_argument(
        "--loss",
        type=str,
        choices=["r1", "wgan", "wgan-gp", "basic"],
        default="r1",
        help="Learning rate decay",
    )
    train_parser.add_argument("--num-epochs", type=int, help="Number of epochs to train (DCGAN)")
    train_parser.add_argument(
        "--save-interval",
        type=int,
        help="Number of epochs to wait before saving models and images",
    )
    train_parser.add_argument(
        "--image-interval",
        type=int,
        help="Number of iterations to wait before saving generated images",
    )
    train_parser.add_argument("--list", action="store_true", help="List available checkpoints")
    train_parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training. Has priority over --checkpoint-epoch",
    )
    train_parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        default=None,
        help="Epoch number of the checkpoint to resume training from",
    )

    # Subparser for the 'create-video' command
    create_video_parser = subparsers.add_parser("create-video", help="Create a video from training frames")
    create_video_parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["DCGAN", "StyleGAN", "VAE"],
        help="Type of model",
    )
    create_video_parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset used")
    create_video_parser.add_argument("--frame-rate", type=int, default=30, help="Frame rate for the video")
    create_video_parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Number of frames to skip between each frame",
    )

    # Subparser for the 'download-all-ffhq' command
    download_all_ffhq_parser = subparsers.add_parser("download-all-ffhq", help="Download all FFHQ images")
    download_all_ffhq_parser.add_argument("--raw", action="store_true", help="Download raw images")
    download_all_ffhq_parser.add_argument("--blur", action="store_true", help="Download blurred images")
    download_all_ffhq_parser.add_argument("--grey", action="store_true", help="Download grayscale images")

    # Subparser for the 'graph-fids' command
    graph_fids_parser = subparsers.add_parser("graph-fids", help="Graph FID scores")
    graph_fids_parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["DCGAN", "StyleGAN", "VAE"],
        help="Type of model",
    )
    graph_fids_parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset used")
    graph_fids_parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        required=True,
        help="Epoch number of the checkpoint to resume training from",
    )

    # Subparser for the 'create-masks' command
    subparsers.add_parser("create-masks", help="Create cutout masks for the images")

    # Subparser for the 'create-blur-and-grey' command
    create_blur_and_grey_parser = subparsers.add_parser("create-blur-and-grey", help="Create blurred and grey datasets")
    create_blur_and_grey_parser.add_argument("--zip", action="store_true", help="Compress the output folders")

    args = parser.parse_args()

    if args.command == "train":
        # Call the train function with all required arguments
        print("\n##############" + "#" * len(f" {args.model} - {args.dataset} ") + "##################")
        print(f"################ {args.model} - {args.dataset} ################")
        print("##############" + "#" * len(f" {args.model} - {args.dataset} ") + "##################\n")
        train_function(
            model=args.model,
            dataset=args.dataset,
            latent_dim=args.latent_dim,
            lr=args.lr,
            loss=args.loss,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            save_interval=args.save_interval,
            image_interval=args.image_interval,
            list_checkpoints_flag=args.list,
            checkpoint_path=args.checkpoint_path,
            checkpoint_epoch=args.checkpoint_epoch,
        )
    elif args.command == "create-video":
        print("\n##############" + "#" * len(f" {args.model} - {args.dataset} ") + "##################")
        print(f"################ {args.model} - {args.dataset} ################")
        print("##############" + "#" * len(f" {args.model} - {args.dataset} ") + "##################\n")
        create_video_function(
            model=args.model,
            dataset=args.dataset,
            frame_rate=args.frame_rate,
            skip_frames=args.skip_frames,
        )
    elif args.command == "download-all-ffhq":
        print("We stored our data on Kaggle.")
        print("Please make sure that you have set up your Kaggle credentials.")

        # If no flag was set, download all three datasets
        if not args.raw and not args.blur and not args.grey:
            print("You are about to download all three datasets.")
            if input("Do you wish to continue (y), or see help on the arguments (n)? (y/n): ").lower() == "y":
                download_all_ffhq(raw=True, blur=True, grey=True)
            else:
                # Print help on the arguments of this command
                parser.parse_args(["download-all-ffhq", "-h"])
        else:
            download_all_ffhq(raw=args.raw, blur=args.blur, grey=args.grey)
    elif args.command == "graph-fids":
        print("\n##############" + "#" * len(f" {args.model} - {args.dataset} ") + "##################")
        print(f"################ {args.model} - {args.dataset} ################")
        print("##############" + "#" * len(f" {args.model} - {args.dataset} ") + "##################\n")
        graph_fids_function(
            model=args.model,
            dataset=args.dataset,
            checkpoint_epoch=args.checkpoint_epoch,
        )
    elif args.command == "create-masks":
        print("Creating masks...")
        create_masks()
    elif args.command == "create-blur-and-grey":
        print("Creating blurred and grey images...")
        create_blur_and_grey(zip_=args.zip)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
