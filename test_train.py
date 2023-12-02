import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch import Trainer, seed_everything

from argparse import ArgumentParser

from src.model.restoration_module import RestorationLitModule
# Here is the place to import your stuff
from src.data.image_datamodule import ImageDataModule as DataModule
from src.model.unet.unet import UNet as Model


def opts_parser():
    usage = "Restores air-images of person in a forest"
    parser = ArgumentParser(description=usage)
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=0.)
    # learning rate + schedule
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    # Data parameters
    parser.add_argument('--data_dirc', type=str)
    # Model parameters
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument('--n_layers', type=int, default=12)
    # Misc.
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument('--experiment_name', type=str, default="Test")
    parser.add_argument('--checkpoint', type=str, default=None)

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
        name=config.experiment_name
    )

    # training dataset loading
    datamodule = DataModule()
    # create pytorch lightening module
    net = Model(4)  # here is the place to init the model(s)
    pl_module = RestorationLitModule(optimizer=AdamW, scheduler=ReduceLROnPlateau, compile=False, encoder=net)
    # create monitor to keep track of learning rate - we want to check the behaviour of our learning rate schedule
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # sets Checkpointing dependency to loss -> we keep the best 2 model according to loss
    checkpoint_callback = ModelCheckpoint(monitor="train/loss", every_n_train_steps=1000, save_last=True, save_top_k=-1)
    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks as well as set up for Mixed Precision
    trainer = L.Trainer(max_epochs=config.n_epochs,
                        logger=wandb_logger,
                        accelerator=device,
                        callbacks=[lr_monitor, checkpoint_callback],
                        default_root_dir=None,  # change dirc if needed
                        precision="32",  # 32 is full precision, maybe we need
                        gradient_clip_val=config.max_grad_norm,  # 0. means no clipping
                        accumulate_grad_batches=1,
                        log_every_n_steps=50)
    # start training
    trainer.fit(pl_module, datamodule=datamodule, ckpt_path=config.checkpoint)

if __name__ == "__main__":
    main()
'''

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
'''