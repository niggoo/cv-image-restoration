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
    parser.add_argument("--images", nargs="+", required=True)  # either 4 images or folder with 4 images
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
        if len(image_files) < 4:
            raise ValueError(f"Found less than 4 images in directory: {args.images[0]}")
        args.images = image_files[:4]  # Select the first 4 images

    return args


def main():
    args = parse_args_and_validate()

    # set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=device)

    # rename state dict keys by removing "model." prefix TODO: why is this (also check if necessary for unet)
    new_state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        new_state_dict[k.replace("model.", "", 1)] = v
    checkpoint["state_dict"] = new_state_dict

    if args.model == "unet" or args.model == "dpt":  # unet and dpt have same forward pass

        if args.model == "unet":
            from src.model.unet.unet import UNet
            model = UNet()
            model.load_state_dict(checkpoint["state_dict"])
            img_standardization = checkpoint["hyper_parameters"]["config"]["img_standardization"]
            mean_ckpt = img_standardization["mean"]
            std_ckpt = img_standardization["std"]

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
            norm_stat = torch.tensor(
                [[107.9049, 30.2373],  # TODO: maybe not hardcode? File is src/data/norm_stats.json
                 [107.9046, 30.2380],
                 [107.9049, 30.2348],
                 [107.9048, 30.2367]])

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
        from PIL import Image

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
    torchvision.utils.save_image(output, args.output)
    print(f"Saved output to: {args.output}")
    # TODO maybe make nice input output comparison plot (see test_plot.py)


if __name__ == "__main__":
    main()
