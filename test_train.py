import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch import Trainer, seed_everything
from torchinfo import summary

import os
from argparse import ArgumentParser

from src.model.restoration_module import RestorationLitModule

# Here is the place to import your stuff
from src.data.emb_datamodule import EmbeddingDataModule as DataModule
from src.model.dinov2.conv_decoder import ModifiedConvHead as Model

# TODO: we should not do this here :D
from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
from test_plot import plot_images


class ImageLoggingCallback(Callback):
    def __init__(self, datamodule, num_samples=5, wandb_logger=None):
        self.datamodule = datamodule
        self.num_samples = num_samples
        self.wandb_logger = wandb_logger
        # save fig to wandb_logger.save_dir
        self.folder_path = os.path.join(self.wandb_logger._project, self.wandb_logger.version)
        # create dir
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

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
                # add filename
                fig_path = os.path.join(self.folder_path, f"val_image_{idx}.png")
                # save fig
                fig.savefig(fig_path)
                

                # Log the image to wandb
                try:
                    self.wandb_logger.log_image(
                        key=f"val_image_{idx}", images=[fig_path]
                    )
                except Exception as e:
                    print(e)
                plt.close(fig)

        pl_module.train()


def opts_parser():
    usage = "Restores air-images of person in a forest"
    parser = ArgumentParser(description=usage)
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    # learning rate + schedule
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    # Data parameters
    parser.add_argument("--data_dirc", type=str)
    # Model parameters
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    # Misc.
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--experiment_name", type=str, default="Test")
    parser.add_argument("--checkpoint", type=str, default=None)

    return parser


def main():
    # parse command line
    parser = opts_parser()
    config = parser.parse_args()
    # set seeds for numpy, torch and python.random
    seed_everything(420, workers=True)
    torch.autograd.set_detect_anomaly(True)
    # training device
    device = "gpu" if torch.cuda.is_available() else "cpu"

    # logging is done using wandb
    wandb_logger = WandbLogger(
        project="CV2023",
        group="Test",
        notes="First Tests",
        config=config,  # this logs all hyperparameters for us
        name=config.experiment_name,
    )

    # training dataset loading
    datamodule = DataModule(data_paths_json_path="src/data/data_paths.json")
    # create pytorch lightening module
    net = Model()  # here is the place to init the model(s)
    print(summary(net))
    pl_module = RestorationLitModule(
        optimizer=AdamW, scheduler=ReduceLROnPlateau, compile=False, encoder=net
    )
    # create monitor to keep track of learning rate - we want to check the behaviour of our learning rate schedule
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # sets Checkpointing dependency to loss -> we keep the best 2 model according to loss
    checkpoint_callback = ModelCheckpoint(
        monitor="train/loss", every_n_train_steps=1000, save_last=True, save_top_k=-1
    )

    image_logging_callback = ImageLoggingCallback(
        datamodule, num_samples=20, wandb_logger=wandb_logger
    )
    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks as well as set up for Mixed Precision
    trainer = L.Trainer(
        max_epochs=config.n_epochs,
        logger=wandb_logger,
        accelerator=device,
        callbacks=[lr_monitor, checkpoint_callback, image_logging_callback],
        default_root_dir=None,  # change dirc if needed
        precision="32",  # 32 is full precision, maybe we need
        gradient_clip_val=config.max_grad_norm,  # 0. means no clipping
        accumulate_grad_batches=config.grad_accum_steps,
        log_every_n_steps=50,
        devices=-1,
    )
    # start training
    trainer.fit(pl_module, datamodule=datamodule, ckpt_path=config.checkpoint)

    # TODO: auto-log some val images + preds to wandb


if __name__ == "__main__":
    main()
"""

sample = next(iter(train_dl)).to("cuda").half()
pl_module.to("cuda")
pl_module.half()
optimizer = torch.optim.Adam(pl_module.model.parameters(), lr=config.learning_rate)
for i in tqdm(range(5000)):
    optimizer.zero_grad()
    noise = torch.randn_like(sample)
    predicted = pl_module.model(sample, torch.tensor(1))
    loss = torch.nn.functional.l1_loss(noise, predicted.squeeze(1))
    loss.backward()
    optimizer.step()
"""
