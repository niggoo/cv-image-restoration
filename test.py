# example usage:
#  python test.py --ckpt /path/to/file.ckpt --output output.png --images image1.png image2.png image3.png image4.png
#  python test.py --ckpt /path/to/file.ckpt --images path/to/folder/with/images
#  python test.py --ckpt /path/to/file.ckpt --mode testset --images path/to/json/file --output path/to/output/folder
# --ckpt: path to checkpoint
# --images:
#   --mode single:
#       must be 4 aos integrated images with focal planes in the order: 0m, -0,5m, -1m, -1,5m or folder with the 4 images
#  --mode testset:
#       path to json file with data paths of the test set
# --output: path for output image(s)

import argparse
import os
import glob
import json
import torch
import torchvision
from torchvision.io import ImageReadMode
#from transformers import AutoImageProcessor, AutoModel
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
#from transformers import logging

#logging.set_verbosity_error()

DINO_SIZE_MAP = {
    "small": 384,
    "base": 768,
    "large": 1024,
}


def parse_args_and_validate():
    valid_models = ["conv_decoder", "unet", "dpt"]
    parser = argparse.ArgumentParser(
        description="""This script is used for inference with different models.
        Focal planes of integral images should be at 0m, -0.5m, -1m, -1.5m in that order and 512x512 pixels.
        Example usage:
        > python test.py --ckpt /path/to/file/model.ckpt --output output.png --images image1.png image2.png image3.png image4.png
        > python test.py --ckpt /path/to/file/model.ckpt --images path/to/folder/with/images
        > python test.py --ckpt /path/to/file/model.ckpt
        > python test.py --ckpt /path/to/model.ckpt --mode testset --images path/to/json/file --output path/to/output/folder""",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # parser.add_argument(
    #     "--model",
    #     type=str,
    #     choices=valid_models,
    #     default="dpt",
    #     help="Model to use for inference (conv_decoder, unet, dpt)",
    # )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./weights/model.ckpt",
        #required=True,
        help="Path to checkpoint corresponding to the model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Path to save the output image, default: ./results",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        help="Mode to run the script in, either single or testset, default: single",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=["./single_input"],
        help="single mode: Either a list of 4 integral images or a directory containing those, default: "
        "./single_input"
        "testset mode: Path to json file containing the data paths of the test set",
    )
    args = parser.parse_args()

    # validate the arguments

    #if args.model not in valid_models:
        #raise ValueError(
            #f"Unknown model: {args.model}, valid models are: {valid_models}"
        #)
    if not os.path.isfile(args.ckpt):
        raise ValueError(f"Checkpoint not found: {args.ckpt}")
    if args.mode not in ["single", "testset"]:
        raise ValueError(
            f"Unknown mode: {args.mode}, valid modes are: single and testset"
        )
    # Check if the provided images argument is a directory
    if args.mode == "single":
        if len(args.images) == 1 and os.path.isdir(args.images[0]):
            # If it's a directory, search for image files in the directory
            image_files = glob.glob(os.path.join(args.images[0], "*.png"))
        else:
            image_files = args.images
        if len(image_files) < 4:
            print(f"Expected 4 images, got {len(image_files)}")
            print("The last image will be used to fill the missing images")
            while len(image_files) < 4:
                image_files.append(image_files[-1])
        elif len(image_files) > 4:
            print("Found more than 4 images")
            print("The first 4 images will be used")
        args.images = image_files[:4]  # Select the first 4 images
    else:
        with open(args.images[0]) as file:
            data_paths = json.load(file)
        args.images = DataLoader(
            dataset=data_paths,
            shuffle=False,
        )

    return args


def plot_output(args, output):
    fig = plt.figure(figsize=(20, 10))
    focal_lengths = [0, 0.5, 1.0, 1.5]
    fig.text(
        0.1, 0.72, "Input", ha="center", va="center", rotation="vertical", fontsize=20
    )

    title = {"conv_decoder": "Convolutional Decoder", "unet": "U-Net", "dpt": "DPT"}

    fig.text(
        0.5, 0.95, f"Model: {title['dpt']}", ha="center", va="center", fontsize=20
    )

    for idx, image_path in enumerate(args.images):
        ax = fig.add_subplot(2, 4, idx + 1)
        ax.imshow(
            Image.open(image_path).resize((512, 512)), cmap="gray", vmin=0, vmax=255
        )
        ax.axis("off")
        ax.set_title(f"Focal length {focal_lengths[idx]}")

    ax = fig.add_subplot(2, 4, 5)
    ax.imshow(output.cpu().numpy().squeeze(), vmin=0, vmax=1, cmap="gray")
    ax.axis("off")
    fig.text(
        0.1, 0.28, "Output", ha="center", va="center", rotation="vertical", fontsize=20
    )
    return plt


def plot_output_testset(integrals, output, gt):
    fig = plt.figure(figsize=(20, 10))
    focal_lengths = [0, 0.5, 1.0, 1.5]
    fig.text(
        0.1, 0.72, "Input", ha="center", va="center", rotation="vertical", fontsize=20
    )

    for idx, image in enumerate(integrals):
        ax = fig.add_subplot(2, 4, idx + 1)
        ax.imshow(image.cpu().numpy().squeeze(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"Focal length {focal_lengths[idx]}")

    ax = fig.add_subplot(2, 4, 5)
    ax.imshow(output.cpu().numpy().squeeze(), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    fig.text(
        0.1, 0.28, "Output", ha="center", va="center", rotation="vertical", fontsize=20
    )

    ax = fig.add_subplot(2, 4, 7)
    ax.imshow(Image.open(gt), cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    fig.text(
        0.5, 0.28, "GT", ha="center", va="center", rotation="vertical", fontsize=20
    )
    return plt


def main():
    args = parse_args_and_validate()

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=device)

    # rename state dict keys by removing "model." prefix
    # because the model was saved using pytorch lightning - slightly different structure to default pytorch
    # lightning adds optimizer and scheduler state dicts to the checkpoint
    checkpoint["state_dict"] = {
        k.replace("model.", "", 1): v for k, v in checkpoint["state_dict"].items()
    }
    print("Using hyperparameters:", checkpoint["hyper_parameters"])

    # single images
    if args.mode == "single":
        #if (
            #args.model == "unet" or args.model == "dpt"
        #):  # unet and dpt have same forward pass
            # if args.model == "unet":
            #     from src.model.unet.unet import UNet

            #     model = UNet()
            #     model.load_state_dict(checkpoint["state_dict"])
            #     img_standardization = checkpoint["hyper_parameters"]["config"][
            #         "img_standardization"
            #     ]
            #     if img_standardization["do_destandardize"] is True:
            #         # load the mean and std from the checkpoint
            #         mean_ckpt = img_standardization["mean"]
            #         std_ckpt = img_standardization["std"]
            #     else:
            #         mean_ckpt = 0
            #         std_ckpt = 1

            #     # load images (same as in src/data/image_datamodule.py)
            #     integral_images = (
            #         torch.stack(
            #             [
            #                 torchvision.io.read_image(image, ImageReadMode.GRAY)
            #                 for image in args.images
            #             ],
            #             dim=0,
            #         )
            #         .squeeze()
            #         .float()
            #     )
            #     # standardize
            #     integral_images = (integral_images - mean_ckpt) / std_ckpt

            #elif args.model == "dpt":
        from src.model.dinov2.dinov2 import Dinov2
        from src.model.dinov2.dpt import DPT

        backbone_size = checkpoint["hyper_parameters"]["config"][
            "backbone_size"
        ]

        dino = Dinov2(
            dinov2_size=backbone_size,
            out_features=checkpoint["hyper_parameters"]["config"][
                "out_features"
            ],
        )
        dpt = DPT(embed_dims=DINO_SIZE_MAP[backbone_size])
        model = torch.nn.Sequential(dino, dpt)

        model.load_state_dict(checkpoint["state_dict"])

        # load images (same as in src/data/dpt_datamodule.py)
        norm_stat = torch.load(os.path.join("src", "data", "norm_stats.pt"))

        norm = torchvision.transforms.Normalize(
            mean=norm_stat[:, 0], std=norm_stat[:, 1]
        )

        print(args.images)
        integral_images = norm(
            torch.stack(
                [
                    torchvision.io.read_image(file, ImageReadMode.GRAY)
                    .float()
                    .squeeze()
                    for file in args.images
                ],
                dim=0,
            )
        )

        # forward pass (unet or dpt)
        integral_images = integral_images.unsqueeze(0).to(device)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            output = model(integral_images)

        # elif args.model == "conv_decoder":
        #     # imports
        #     from src.model.dinov2.conv_decoder import ModifiedConvHead

        #     # load the dino
        #     backbone_size = checkpoint["hyper_parameters"]["config"]["backbone_size"]
        #     model_name = f"facebook/dinov2-{backbone_size}"
        #     dino = AutoModel.from_pretrained(model_name)
        #     # load the conv decoder
        #     conv_decoder = ModifiedConvHead(in_channels=DINO_SIZE_MAP[backbone_size])
        #     conv_decoder.load_state_dict(checkpoint["state_dict"])

        #     # load the images (same as in src/model/dinov2/dino/compute_embeddings.py)
        #     processor = AutoImageProcessor.from_pretrained(model_name)
        #     processor.crop_size = {"height": 512, "width": 512}
        #     processor.size["shortest_edge"] = 512

        #     # generate embeddings for each image using dino
        #     dino.to(device=device)
        #     embeddings = []
        #     for image_path in args.images:
        #         image = Image.open(image_path)
        #         inputs = processor(images=image, return_tensors="pt")
        #         inputs.to(device=device)
        #         outputs = dino(**inputs)
        #         embeddings.append(
        #             outputs.last_hidden_state[:, 1:, :]
        #         )  # remove CLS token and keep only last hidden state

        #     # stack the embeddings
        #     embeddings = torch.stack(embeddings, dim=1).to(device=device)

        #     # forward pass (conv_decoder)
        #     conv_decoder = conv_decoder.to(device=device)
        #     conv_decoder.eval()
        #     with torch.no_grad():
        #         output = conv_decoder(embeddings).squeeze(-1)

        # threshold the output
        # output = sharpen_filter(output)
        # output = threshold_func(output, 0.7)

        # save the output
        # torchvision.utils.save_image(output, args.output)
        plt = plot_output(args, output)
        # plt.show()
        outpath = os.path.join(args.output, "output.png")
        plt.savefig(outpath)
        print(f"Saved output to: {args.output}")
        # also save single output image
        torchvision.utils.save_image(output, outpath.replace(".png", "_single.png"))

    elif args.mode == "testset":
        if "png" in args.output:
            args.output = args.output.replace(".png", "")
        # if args.model == "unet":
        #     from src.model.unet.unet import UNet

        #     model = UNet()
        #     model.load_state_dict(checkpoint["state_dict"])
        #     img_standardization = checkpoint["hyper_parameters"]["config"][
        #         "img_standardization"
        #     ]
        #     if img_standardization["do_destandardize"] is True:
        #         # load the mean and std from the checkpoint
        #         mean_ckpt = img_standardization["mean"]
        #         std_ckpt = img_standardization["std"]
        #     else:
        #         mean_ckpt = 0
        #         std_ckpt = 1

        #     for imgs in tqdm(args.images):
        #         # skip images that dont have 4 focal planes
        #         if len(imgs["integral_images"]) != 4:
        #             continue
        #         # load images (same as in src/data/image_datamodule.py)
        #         integral_images = (
        #             torch.stack(
        #                 [
        #                     torchvision.io.read_image(image[0], ImageReadMode.GRAY)
        #                     for image in imgs["integral_images"]
        #                 ],
        #                 dim=0,
        #             )
        #             .squeeze()
        #             .float()
        #         )
        #         # standardize
        #         integral_images = (integral_images - mean_ckpt) / std_ckpt

        #         # forward pass (unet or dpt)
        #         integral_images = integral_images.unsqueeze(0).to(device)
        #         model = model.to(device)
        #         model.eval()
        #         with torch.no_grad():
        #             output = model(integral_images)

        #         plt = plot_output_testset(
        #             integral_images.squeeze(0), output, imgs["GT"][0]
        #         )
        #         # plt.show()
        #         outpath = os.path.join(args.output, imgs["batch"][0])
        #         if not os.path.exists(outpath):
        #             os.makedirs(outpath)
        #         outpath = os.path.join(outpath, imgs["image_id"][0]) + "_full.png"
        #         plt.savefig(outpath)
        #         plt.close()
        #         # also save single output image
        #         torchvision.utils.save_image(output, outpath.replace("full", "single"))

        #elif args.model == "dpt":
        from src.model.dinov2.dinov2 import Dinov2
        from src.model.dinov2.dpt import DPT

        backbone_size = checkpoint["hyper_parameters"]["config"]["backbone_size"]

        dino = Dinov2(
            dinov2_size=backbone_size,
            out_features=checkpoint["hyper_parameters"]["config"]["out_features"],
        )
        dpt = DPT(embed_dims=DINO_SIZE_MAP[backbone_size])
        model = torch.nn.Sequential(dino, dpt)

        model.load_state_dict(checkpoint["state_dict"])

        # load images (same as in src/data/dpt_datamodule.py)
        norm_stat = torch.load(os.path.join("src", "data", "norm_stats.pt")).to(
            device
        )

        norm = torchvision.transforms.Normalize(
            mean=norm_stat[:, 0], std=norm_stat[:, 1]
        )
        for imgs in tqdm(args.images):
            # skip images that dont have 4 focal planes
            if len(imgs["integral_images"]) != 4:
                continue
            integral_images = norm(
                torch.stack(
                    [
                        torchvision.io.read_image(image[0], ImageReadMode.GRAY)
                        for image in imgs["integral_images"]
                    ],
                    dim=0,
                )
                .squeeze()
                .float()
            )

            # forward pass (unet or dpt)
            integral_images = integral_images.unsqueeze(0).to(device)
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                output = model(integral_images)

            # do the inverse normalization on the integral images
            integral_images = integral_images * norm_stat[:, 1].view(
                4, 1, 1
            ) + norm_stat[:, 0].view(4, 1, 1)
            integral_images = integral_images.squeeze(0) / 255
            plt = plot_output_testset(
                integral_images,
                output,
                imgs["GT"][0],
            )
            # plt.show()
            outpath = os.path.join(args.output, imgs["batch"][0])
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            outpath = os.path.join(outpath, imgs["image_id"][0]) + "_full.png"
            plt.savefig(outpath)
            plt.close()
            # also save single output image
            torchvision.utils.save_image(output, outpath.replace("full", "single"))

        # elif args.model == "conv_decoder":
        #     # imports
        #     from src.model.dinov2.conv_decoder import ModifiedConvHead

        #     # load the dino
        #     backbone_size = checkpoint["hyper_parameters"]["config"]["backbone_size"]
        #     model_name = f"facebook/dinov2-{backbone_size}"
        #     dino = AutoModel.from_pretrained(model_name)
        #     # load the conv decoder
        #     conv_decoder = ModifiedConvHead(in_channels=DINO_SIZE_MAP[backbone_size])
        #     conv_decoder.load_state_dict(checkpoint["state_dict"])

        #     # load the images (same as in src/model/dinov2/dino/compute_embeddings.py)
        #     processor = AutoImageProcessor.from_pretrained(model_name)
        #     processor.crop_size = {"height": 512, "width": 512}
        #     processor.size["shortest_edge"] = 512

        #     # generate embeddings for each image using dino
        #     dino.to(device=device)
        #     conv_decoder.to(device=device)

        #     for imgs in tqdm(args.images):
        #         # skip images that dont have 4 focal planes
        #         if len(imgs["integral_images"]) != 4:
        #             continue
        #         embeddings = []
        #         for image_path in imgs["integral_images"]:
        #             image = Image.open(image_path[0])
        #             inputs = processor(images=image, return_tensors="pt")
        #             inputs.to(device=device)
        #             outputs = dino(**inputs)
        #             embeddings.append(
        #                 outputs.last_hidden_state[:, 1:, :]
        #             )  # remove CLS token and keep only last hidden state

        #         integral_images = (
        #             torch.stack(
        #                 [
        #                     torchvision.io.read_image(image[0], ImageReadMode.GRAY)
        #                     for image in imgs["integral_images"]
        #                 ],
        #                 dim=0,
        #             )
        #             .squeeze()
        #             .float()
        #         )

        #         # stack the embeddings
        #         embeddings = torch.stack(embeddings, dim=1).to(device=device)
        #         # forward pass (conv_decoder)
        #         conv_decoder = conv_decoder.to(device=device)
        #         conv_decoder.eval()
        #         with torch.no_grad():
        #             output = conv_decoder(embeddings).squeeze(-1)

        #         plt = plot_output_testset(integral_images, output, imgs["GT"][0])
        #         # plt.show()
        #         outpath = os.path.join(args.output, imgs["batch"][0])
        #         if not os.path.exists(outpath):
        #             os.makedirs(outpath)
        #         outpath = os.path.join(outpath, imgs["image_id"][0]) + "_full.png"
        #         plt.savefig(outpath)
        #         plt.close()
        #         # also save single output image
        #         torchvision.utils.save_image(output, outpath.replace("full", "single"))


if __name__ == "__main__":
    main()
