from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from einops import rearrange
from ..utils.metrics import SILogLoss, ScaleAndShiftInvariantLoss, DiscreteNLLLoss


class RestorationLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        encoder: torch.nn.Module = torch.nn.Identity(),
        decoder: torch.nn.Module = torch.nn.Identity(),
    ) -> None:
        """
        :param encoder: Encoder model to train defaults to identity if not needed
        :param decoder: Decoder model to train defaults to identity if not needed
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # ignore takes out the torch modules, otherwise they are saved twice
        self.save_hyperparameters(
            logger=False, ignore=["encoder", "decoder"]
        )  # True --> we additionally log the hyperparameter

        self.encoder = encoder
        self.decoder = decoder

        # loss function
        self.loss_fn = nn.MSELoss()

        # metric objects for calculating and averaging across batches
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        self.test_psnr = PeakSignalNoiseRatio(data_range=1)
        self.test_ssim = StructuralSimilarityIndexMeasure(data_range=1)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation Metric
        self.val_psnr_best = MaxMetric()
        self.val_ssim_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the models

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_psnr.reset()
        self.val_psnr_best.reset()
        self.val_ssim.reset()
        self.val_ssim_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and ground truth.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of ground truth.
        """
        x, gt = batch
        restored = self.forward(x)
        # check for nan values
        if torch.isnan(restored).any():
            print("Nan values in prediction")
            # replace with zeros
            restored[torch.isnan(restored)] = 0
            raise ValueError("Nan values in prediction")
        # restored = restored + gt
        # mask = torch.where(
        #     gt == 0, torch.tensor(0.0).to(gt.device), torch.tensor(1.0).to(gt.device)
        # ).to(gt.device).bool()
        loss = self.loss_fn(restored, gt)
        return loss, restored, gt

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        mloss = self.train_loss(loss)
        mpsnr = self.train_psnr(preds, targets)
        ssim = self.train_ssim(self.rearrange_for_ssim(preds), self.rearrange_for_ssim(targets))

        self.log("train/loss", mloss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/psnr", mpsnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/ssim", ssim, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        mloss = self.val_loss(loss)
        mpsnr = self.val_psnr(preds, targets)
        ssim = self.val_ssim(self.rearrange_for_ssim(preds), self.rearrange_for_ssim(targets))

        self.log("val/loss", mloss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/psnr", mpsnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ssim", ssim, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        psnr = self.val_psnr.compute()  # get current val psnr
        self.val_psnr_best(psnr)  # update best so far val psnr
        ssim = self.val_ssim.compute()  # get current val ssim
        self.val_ssim_best(ssim)  # update best so far val ssim
        # log `val_psnr_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/psnr_best", self.val_psnr_best.compute(), sync_dist=True, prog_bar=True
        )
        self.log(
            "val/ssim_best", self.val_ssim_best.compute(), sync_dist=True, prog_bar=True
        )
        

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        mloss = self.train_test(loss)
        mpsnr = self.train_test(preds, targets)
        ssim = self.test_ssim(self.rearrange_for_ssim(preds), self.rearrange_for_ssim(targets))

        self.log("test/loss", mloss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/psnr", mpsnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ssim", ssim, on_step=False, on_epoch=True, prog_bar=True)

    def rearrange_for_ssim(self, data):
        if data.shape[3] == 1:
            return rearrange(data, "b h w c -> b c h w")
        else:
            return data
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(
                optimizer=optimizer
            )  # with this we can do some fancy lr scheduling
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = RestorationLitModule(None, None, None)
