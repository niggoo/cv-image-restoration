import torch
import json
from lightning.pytorch import seed_everything
from src.model.restoration_module import RestorationLitModule
from pathlib import Path

# Here is the place to import your stuff
from src.data.emb_datamodule import EmbeddingDataModule as DataModule
from src.model.dinov2.simple_decoder import SimpleDecoder as Model


def plot_images(raw, pred, gt):
    import matplotlib.pyplot as plt

    raw = raw * 255
    gt = gt * 255
    # scale prediction values to [0, 255] using min-max scaling
    # pred = pred - pred.min()
    # pred = pred / pred.max()
    pred = pred * 255
    raw = raw.numpy().astype(int)
    gt = gt.numpy().astype(int)
    pred = pred.cpu().numpy().astype(int)
    cmap = "gray"
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(raw, cmap=cmap)
    axs[0].set_title("Input")
    axs[1].imshow(pred, cmap=cmap)
    axs[1].set_title("Prediction")
    axs[2].imshow(gt, cmap=cmap)
    axs[2].set_title("GT")
    plt.tight_layout()
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig


def main():
    seed_everything(420, workers=True)

    # load random data
    data_paths_json_path = "src/data/data_paths.json"
    with open(data_paths_json_path) as file:
        data_paths = json.load(file)

    # load model
    net = Model()
    net.eval()
    # load data
    data_module = DataModule(data_paths_json_path=data_paths_json_path)
    data_module.setup()
    # load checkpoint
    pl_module = RestorationLitModule(
        optimizer=None, scheduler=None, compile=None, encoder=net
    )
    checkpoint_path = "CV2023/al7538jh/checkpoints/last.ckpt"
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


if __name__ == "__main__":
    main()
