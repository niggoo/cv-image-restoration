# example usage:
#  python test.py --model conv_decoder --ckpt /path/to/file.ckpt --output output.png --images image1.png image2.png image3.png image4.png
#  python test.py --model unet --ckpt /path/to/file.ckpt --images path/to/folder/with/images
# options for --model: conv_decoder, unet, dpt
# --ckpt: path to checkpoint
# --images: must be 4 aos integrated images with focal planes in the order: 0m, -0,5m, -1m, -1,5m or folder with the 4 images
# --output: path for output image

import argparse
import os
import glob
import torch
import torchvision
from torchvision.io import ImageReadMode
from transformers import AutoImageProcessor, AutoModel
import matplotlib.pyplot as plt
from PIL import Image

from transformers import logging

logging.set_verbosity_error()

DINO_SIZE_MAP = {
    "small": 384,
    "base": 768,
    "large": 1024,
}


def parse_args_and_validate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--images", nargs="+", default=["real_focal_stack"])  # either 4 images or folder with 4 images
    args = parser.parse_args()

    # validate the arguments
    valid_models = ["conv_decoder", "unet", "dpt"]
    if args.model not in valid_models:
        raise ValueError(f"Unknown model: {args.model}, valid models are: {valid_models}")
    if not os.path.isfile(args.ckpt):
        raise ValueError(f"Checkpoint not found: {args.ckpt}")
    # Check if the provided images argument is a directory
    if len(args.images) == 1 and os.path.isdir(args.images[0]):
        # If it's a directory, search for image files in the directory
        image_files = glob.glob(os.path.join(args.images[0], '*.png'))
    else:
        image_files = args.images
    if len(image_files) < 4:
        print(f"Expected 4 images, got {len(image_files)}")
        print("The last image will be used to fill the missing images")
        while len(image_files) < 4:
            image_files.append(image_files[-1])
    elif len(image_files) > 4:
        print(f"Found more than 4 images")
        print("The first 4 images will be used")
    args.images = image_files[:4]  # Select the first 4 images

    return args


def plot_output(args, output):
    fig = plt.figure(figsize=(20, 10))
    focal_lengths = [0, 0.5, 1.0, 1.5]
    fig.text(0.1, 0.72, 'Input', ha='center', va='center', rotation='vertical', fontsize=20)

    title = {"conv_decoder": "Convolutional Decoder", "unet": "U-Net", "dpt": "DPT"}

    fig.text(0.5, 0.95, f'Model: {title[args.model]}', ha='center', va='center', fontsize=20)

    for idx, image_path in enumerate(args.images):
        ax = fig.add_subplot(2, 4, idx + 1)
        ax.imshow(Image.open(image_path).resize((512, 512)))
        ax.axis("off")
        ax.set_title(f'Focal length {focal_lengths[idx]}')

    ax = fig.add_subplot(2, 4, 5)
    ax.imshow(output.cpu().numpy().squeeze(), cmap="gray")
    ax.axis("off")
    fig.text(0.1, 0.28, 'Output', ha='center', va='center', rotation='vertical', fontsize=20)
    return plt


def main():
    args = parse_args_and_validate()

    # set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=device)

    # rename state dict keys by removing "model." prefix 
    # because the model was saved using pytorch lightning - slightly different structure to default pytorch
    # lightning adds optimizer and scheduler state dicts to the checkpoint
    checkpoint["state_dict"] = {k.replace("model.", "", 1): v for k, v in checkpoint["state_dict"].items()}
    #

    if args.model == "unet" or args.model == "dpt":  # unet and dpt have same forward pass

        if args.model == "unet":
            from src.model.unet.unet import UNet
            model = UNet()
            model.load_state_dict(checkpoint["state_dict"])
            img_standardization = checkpoint["hyper_parameters"]["config"]["img_standardization"]
            if img_standardization["do_destandardize"] is True:
                # load the mean and std from the checkpoint
                mean_ckpt = img_standardization["mean"]
                std_ckpt = img_standardization["std"]
            else:
                mean_ckpt = 0
                std_ckpt = 1

            # load images (same as in src/data/image_datamodule.py)
            integral_images = (
                torch.stack(
                    [torchvision.io.read_image(image, ImageReadMode.GRAY) for image in args.images],
                    dim=0, ).squeeze().float())
            # standardize
            integral_images = (integral_images - mean_ckpt) / std_ckpt

        elif args.model == "dpt":
            from src.model.dinov2.dinov2 import Dinov2
            from src.model.dinov2.dpt import DPT
            backbone_size = checkpoint["hyper_parameters"]["config"]["backbone_size"]

            dino = Dinov2(dinov2_size=backbone_size,
                          out_features=checkpoint["hyper_parameters"]["config"]["out_features"])
            dpt = DPT()
            model = torch.nn.Sequential(dino, dpt)

            model.load_state_dict(checkpoint["state_dict"])

            # load images (same as in src/data/dpt_datamodule.py)
            norm_stat = torch.load(os.path.join("src", "data", "norm_stats.pt"))

            norm = torchvision.transforms.Normalize(mean=norm_stat[:, 0], std=norm_stat[:, 1])
            integral_images = norm(
                torch.stack(
                    [torchvision.io.read_image(file, ImageReadMode.GRAY).float().squeeze() for file in args.images],
                    dim=0)
            )

        # forward pass (unet or dpt)
        integral_images = integral_images.unsqueeze(0).to(device)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            output = model(integral_images)

    elif args.model == "conv_decoder":
        # imports
        from src.model.dinov2.conv_decoder import ModifiedConvHead

        # load the dino
        backbone_size = checkpoint["hyper_parameters"]["config"]["backbone_size"]
        model_name = f"facebook/dinov2-{backbone_size}"
        dino = AutoModel.from_pretrained(model_name)
        # load the conv decoder
        conv_decoder = ModifiedConvHead(in_channels=DINO_SIZE_MAP[backbone_size])
        conv_decoder.load_state_dict(checkpoint["state_dict"])

        # load the images (same as in src/model/dinov2/dino/compute_embeddings.py)
        processor = AutoImageProcessor.from_pretrained(model_name)
        processor.crop_size = {"height": 512, "width": 512}
        processor.size["shortest_edge"] = 512

        # generate embeddings for each image using dino
        dino.to(device=device)
        embeddings = []
        for image_path in args.images:
            image = Image.open(image_path)
            inputs = processor(images=image, return_tensors="pt")
            inputs.to(device=device)
            outputs = dino(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 1:, :])  # remove CLS token and keep only last hidden state

        # stack the embeddings
        embeddings = torch.stack(embeddings, dim=1).to(device=device)

        # forward pass (conv_decoder)
        conv_decoder = conv_decoder.to(device=device)
        conv_decoder.eval()
        with torch.no_grad():
            output = conv_decoder(embeddings).squeeze(-1)

    # save the output
    # torchvision.utils.save_image(output, args.output)
    plt = plot_output(args, output)
    # plt.show()
    plt.savefig(args.output)
    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()
    # python test.py --model conv_decoder --ckpt cktpts/epoch=63-step=7296_DINO.ckpt --output real_integrals_results/conv_decoder_result.png
    # python test.py --model dpt --ckpt cktpts/epoch=147-step=105376_DPT.ckpt --output real_integrals_results/dpt_result.png
    # python test.py --model unet --ckpt cktpts/epoch=36-step=52688_UNET.ckpt --output real_integrals_results/unet_result.png
