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
from src.data.hf_datamodule import HFImageDataModule
from src.data.image_datamodule import ImageDataModule
from src.model.dinov2.conv_decoder import ModifiedConvHead
from src.model.restoration_module import RestorationLitModule
from src.model.unet.unet import UNet
from test_plot import plot_images, plot_pred_image

DINO_SIZE_MAP = {
    "small": 384,
    "base": 768,
    "large": 1024,
}


@hydra.main(version_base=None, config_path="./configs/", config_name="dino")
def main(cfg: DictConfig):
    config = cfg
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
    # create monitor to keep track of learning rate - we want to check the behaviour of our learning rate schedule
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # sets Checkpointing dependency to loss -> we keep the best 2 model according to loss
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss", every_n_epochs=1, save_last=True, save_top_k=1
    )

    image_logging_callback = ImageLoggingCallback(
        datamodule, num_samples=100, wandb_logger=wandb_logger
    )
    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        min_delta=config.early_stopping.min_delta,
        patience=config.early_stopping.patience,
        verbose=True,
        mode="min",
        log_rank_zero_only=True,
    )

    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks as well as set up for Mixed Precision
    trainer = L.Trainer(
        max_epochs=config.n_epochs,
        logger=wandb_logger,
        callbacks=[
            lr_monitor,
            checkpoint_callback,
            image_logging_callback,
            early_stop_callback,
        ],
        default_root_dir=None,  # change dirc if needed
        precision="32",  # 32 is full precision, maybe we need
        gradient_clip_val=config.max_grad_norm,  # 0. means no clipping
        accumulate_grad_batches=config.grad_accum_steps,
        log_every_n_steps=50,
        accelerator=config.device.type or device,
        devices=config.device.ids or None,
    )
    # start training
    trainer.fit(pl_module, datamodule=datamodule, ckpt_path=config.checkpoint)

    # TODO: optionally, test the model
    # FIXME @Noah!
    # but how to do this best with lightning?


def init_wandb(config):
    logger = WandbLogger(
        project=config.wandb.project,
        group=config.wandb.group,
        notes=config.wandb.notes,
        config=OmegaConf.to_container(config),  # this logs all hyperparameters for us
        name=config.wandb.experiment_name,
    )
    return logger


def get_model(config):
    # TODO: add possibility to add hyperparams here! like Conv in_channels, etc.
    if config.model == "UNet":
        return UNet()
    elif config.model == "ModifiedConvHead":
        return ModifiedConvHead(in_channels=DINO_SIZE_MAP[config.backbone_size])


class ImageLoggingCallback(Callback):
    def __init__(self, datamodule, num_samples=5, wandb_logger=None):
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
                    # sometimes, there are weird errors
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
        # XXX: difference betwen ImageDataModule and HFImageDataModule:
        # HF uses HuggingFace a pretrained model and normalizes using a processor
        # ImageDataModule uses the provided mean and std to normalize, calculated on the train set
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
    elif config.datamodule == "EmbeddingDataModule":
        return EmbeddingDataModule(
            data_paths_json_path=f"src/data/data_paths_{config.backbone_size}.json",
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            data_limit=data_limit,
            oversample=config.data.oversample,
        )
    elif config.datamodule == "HFImageDataModule":
        return HFImageDataModule(
            backbone_model_name=config.backbone_model_name,
            data_paths_json_path="src/data/data_paths.json",
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            data_limit=data_limit,
            oversample=config.data.oversample,        )
    else:
        raise ValueError(f"Unknown datamodule parameter given: {config.datamodule}!")


if __name__ == "__main__":
    main()
    # some example commands:
    # python3 test_train.py config-name=unet
    # python3 test_train.py config-name=unet img_standardization.mean=100 learning_rate=0.0001
    # python3 test_train.py --config-name=dino
    # python3 test_train.py --config-name=dino 'wandb.experiment_name=Small 1e-3, oversample'
