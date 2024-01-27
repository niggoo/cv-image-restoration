import os
import sys

import hydra
import lightning as L
import matplotlib.pyplot as plt
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary

from src.data.emb_datamodule import EmbeddingDataModule
from src.data.image_datamodule import ImageDataModule
from src.data.dpt_datamodule import DptImageDataModule
from src.model.dinov2.conv_decoder import ModifiedConvHead
from src.model.restoration_module import RestorationLitModule
from src.model.unet.unet import UNet
from src.model.dinov2.dinov2 import Dinov2
from src.model.dinov2.dpt import DPT
from src.utils.test_plot import plot_images, plot_pred_image

DINO_SIZE_MAP = {
    "small": 384,
    "base": 768,
    "large": 1024,
}


@hydra.main(version_base=None, config_path="./src/configs/", config_name="dino-dpt")
def main(config: DictConfig):
    # set seeds for numpy, torch and python.random
    seed_everything(420, workers=True)
    torch.autograd.set_detect_anomaly(True)
    # training device
    device = "gpu" if torch.cuda.is_available() else "cpu"

    # logging is done using wandb
    wandb_logger = init_wandb(config)

    # training dataset loading
    datamodule = get_datamodule(config)
    # create pytorch lightening module
    net = get_model(config)

    print(summary(net))

    pl_module = RestorationLitModule(
        optimizer=AdamW,
        scheduler=ReduceLROnPlateau,
        compile=False,
        config=config,
        model=net,
    )
    callbacks = []

    # create monitor to keep track of learning rate - we want to check the behaviour of our learning rate schedule
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # sets Checkpointing dependency to loss -> we keep the best 2 model according to loss
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss", every_n_epochs=1, save_last=True, save_top_k=1
    )
    callbacks.append(checkpoint_callback)

    if config.logging:
        image_logging_callback = ImageLoggingCallback(
            datamodule, num_samples=100, wandb_logger=wandb_logger
        )
        callbacks.append(image_logging_callback)

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        min_delta=config.early_stopping.min_delta,
        patience=config.early_stopping.patience,
        verbose=True,
        mode="min",
        log_rank_zero_only=True,
    )
    callbacks.append(early_stop_callback)

    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks as well as set up for Mixed Precision
    trainer = L.Trainer(
        max_epochs=config.n_epochs,
        logger=wandb_logger if config.logging else None,
        callbacks=callbacks,
        default_root_dir=None,  # change dirc if needed
        precision=config.precision if hasattr(config, "precision") else "32",
        gradient_clip_val=config.max_grad_norm,  # 0. means no clipping
        accumulate_grad_batches=config.grad_accum_steps,
        log_every_n_steps=50,
        accelerator=config.device.type or device,
        devices=config.device.ids or None,
        num_sanity_val_steps=0,
    )
    # start training
    trainer.fit(pl_module, datamodule=datamodule, ckpt_path=config.checkpoint)

    # test trained model
    trainer.test(datamodule=datamodule, ckpt_path="best")


def init_wandb(config):
    logger = WandbLogger(
        project=config.wandb.project,
        entity=config.wandb.entity,
        group=config.wandb.group,
        notes=config.wandb.notes,
        config=OmegaConf.to_container(config),  # this logs all hyperparameters for us
        name=config.wandb.experiment_name,
    )

    return logger


def get_model(config):
    if config.model == "UNet":
        return UNet()
    elif config.model == "ModifiedConvHead":
        return ModifiedConvHead(in_channels=DINO_SIZE_MAP[config.backbone_size])
    elif config.model == "DPT":
        dino = Dinov2(
            dinov2_size=config.backbone_size,
            out_features=config.out_features,
            freeze_encoder=config.freeze_encoder,
            skip=config.skip,
        )
        dpt = DPT(embed_dims=DINO_SIZE_MAP[config.backbone_size])
        return torch.nn.Sequential(dino, dpt)


class ImageLoggingCallback(Callback):
    """Callback to log images to wandb during training."""

    def __init__(self, datamodule, num_samples=5, wandb_logger=None):
        """Initialize the callback.

        Args:
            datamodule (pl.LightningDataModule): datamodule to get the images from
            num_samples (int, optional): How many samples to log. Defaults to 5.
            wandb_logger: The main wandb logger
        """
        self.datamodule = datamodule
        self.num_samples = num_samples
        self.wandb_logger = wandb_logger
        # save fig to wandb_logger.save_dir
        self.folder_path = os.path.join(
            self.wandb_logger._project, self.wandb_logger.version
        )
        # create dir
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        if not os.path.exists(os.path.join(self.folder_path, "single")):
            os.makedirs(os.path.join(self.folder_path, "single"))

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        with torch.no_grad():
            for idx in range(self.num_samples):
                embeddings, gt, raw, _ = self.datamodule.data_val.get_all(idx)
                # print pl module device
                embeddings = embeddings.to(pl_module.device)
                # add batch dimension
                pred = pl_module(embeddings.unsqueeze(0))
                pred = pred.squeeze(0).detach()

                fig = plot_images(raw=raw, pred=pred, gt=gt)
                fig_single = plot_pred_image(pred)
                # add filename
                fig_path = os.path.join(self.folder_path, f"val_image_{idx}.png")
                fig_single_path = os.path.join(
                    self.folder_path, "single", f"val_image_{idx}.png"
                )

                # save figs
                fig.savefig(fig_path)
                fig_single.savefig(fig_single_path)

                # Log the image to wandb
                try:
                    self.wandb_logger.log_image(
                        key=f"val_image_{idx}", images=[fig_path]
                    )
                except Exception as e:
                    # sometimes, there are errors due to multi-processing
                    print(e)
                # keep independent to log as many images as possible
                try:
                    self.wandb_logger.log_image(
                        key=f"val_image_{idx}_single", images=[fig_single_path]
                    )
                except Exception as e:
                    print(e)
                plt.close(fig)
                plt.close(fig_single)

        pl_module.train()


def get_datamodule(config):
    data_limit = config.data.limit if config.data.limit is not None else sys.maxsize
    if config.datamodule == "ImageDataModule":
        # ImageDataModule uses the provided mean and std to normalize, calculated on the train set
        # uses full images (for UNet Baseline)
        return ImageDataModule(
            mean=config.img_standardization.mean
            if config.img_standardization.do_destandardize
            else 0,
            std=config.img_standardization.std
            if config.img_standardization.do_destandardize
            else 1,
            data_paths_json_path="src/data/data_paths.json",
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            data_limit=data_limit,
            oversample=config.data.oversample,
        )
    # uses pre-computed embeddings (using DINOv2 + Conv-Decoder)
    elif config.datamodule == "EmbeddingDataModule":
        return EmbeddingDataModule(
            data_paths_json_path=f"src/data/data_paths_{config.backbone_size}.json",
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            data_limit=data_limit,
            oversample=config.data.oversample,
        )
    # uses on-the fly embeddings (using DINOv2 + DPT)
    elif config.datamodule == "DPTImageDataModule":
        return DptImageDataModule(
            data_paths_json_path="src/data/data_paths.json",
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            data_limit=data_limit,
            oversample=config.data.oversample,
            augment=config.data.augment,
        )
    else:
        raise ValueError(f"Unknown datamodule parameter given: {config.datamodule}!")


if __name__ == "__main__":
    main()
    # some example commands:
    # python3 train.py config-name=unet
    # python3 train.py config-name=unet img_standardization.mean=100 learning_rate=0.0001
    # python3 train.py --config-name=dino
    # python3 train.py --config-name=dino 'wandb.experiment_name=Small 1e-3, oversample'