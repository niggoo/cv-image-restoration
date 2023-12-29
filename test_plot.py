import torch
import json
import matplotlib.pyplot as plt

from lightning.pytorch import seed_everything
from src.model.restoration_module import RestorationLitModule
from pathlib import Path

from src.data.emb_datamodule import EmbeddingDataModule as DataModule
from src.model.dinov2.conv_decoder import ModifiedConvHead as Model


def plot_images(raw, pred, gt):
    raw = raw * 255
    gt = gt * 255
    # scale prediction values to [0, 255] using min-max scaling
    # TODO: is this needed for any of our configs?
    # pred = pred - pred.min()
    # pred = pred / pred.max()
    pred = pred * 255
    raw = raw.cpu().numpy().astype(int)
    gt = gt.cpu().numpy().astype(int)
    pred = pred.cpu().numpy().astype(int)
    cmap = "gray"
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(raw.reshape(512, 512), cmap=cmap)
    axs[0].set_title("Input")
    axs[1].imshow(pred.reshape(512, 512), cmap=cmap)
    axs[1].set_title("Prediction")
    axs[2].imshow(gt.reshape(512, 512), cmap=cmap)
    axs[2].set_title("GT")
    plt.tight_layout()
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig


def plot_pred_image(pred):
    # scale prediction values to [0, 255] using min-max scaling
    # pred = pred - pred.min()
    # pred = pred / pred.max()
    pred = pred * 255
    pred = pred.cpu().numpy().astype(int)
    cmap = "gray"

    # Setting figure size to match the image size
    # XXX: this assumes images are 512x512!
    fig_size = 512 / 80  # Convert pixels to inches for matplotlib (80 dpi is default)
    fig = plt.figure(figsize=(fig_size, fig_size), dpi=80, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1], xticks=[], yticks=[], frame_on=False)
    ax.imshow(pred.reshape(512, 512), cmap=cmap)
    ax.axis("off")

    return fig


def main():
    seed_everything(420, workers=True)

    # load random data
    data_paths_json_path = "src/data/data_paths.json"
    with open(data_paths_json_path) as file:
        data_paths = json.load(file)

    # load model
    net = Model(in_channels=768)
    net.eval()
    # load data
    data_module = DataModule(data_paths_json_path=data_paths_json_path)
    data_module.setup()
    # load checkpoint
    pl_module = RestorationLitModule(
        optimizer=None, scheduler=None, compile=None, encoder=net
    )
    checkpoint_path = "CV2023/v8d05buj/checkpoints/epoch=7-step=22000.ckpt"
    checkpoint = torch.load(checkpoint_path)
    pl_module.load_state_dict(checkpoint["state_dict"])

    for idx in range(30):
        embeddings, gt, raw, params = data_module.data_test.get_all(idx)
        # get prediction
        pred = pl_module(embeddings.unsqueeze(0))
        pred = pred.squeeze(0).detach()
        # plot
        fig = plot_images(raw=raw, pred=pred, gt=gt)
        # create folder in checkpoint path to save the images
        Path(checkpoint_path).parent.joinpath("images").mkdir(exist_ok=True)
        # save
        fig.savefig(
            Path(checkpoint_path)
            .parent.joinpath("images")
            .joinpath(f"image_{idx}_test.png")
        )
        fig.show()
