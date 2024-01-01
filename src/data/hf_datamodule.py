import sys

import torch

from src.data.base_datamodule import BaseDataModule
import json
from typing import Optional, Tuple
import random
import torchvision
from torch.utils.data import Dataset, random_split
from torchvision.io import ImageReadMode
from transformers import AutoImageProcessor


class HFDataSet(Dataset):
    def __init__(self, data_paths: dict, backbone_model_name: str = "facebook/dpt-dinov2-small-nyu",
                 data_limit: int = sys.maxsize):
        super().__init__()
        self.data_paths = data_paths
        self.processor = AutoImageProcessor.from_pretrained(backbone_model_name)
        self.data_limit = data_limit
        # self.processor.crop_size = {"height": 512, "width": 512}

    def __len__(self):
        return min(len(self.data_paths), self.data_limit)

    def get_all(self, idx):
        emb, gt = self.__getitem__(idx)
        # load the raw image
        raw = torchvision.io.read_image(
            self.data_paths[idx]["raw_images"][0], ImageReadMode.GRAY
        ).float()
        raw = raw.permute(1, 2, 0)  # HWC
        # load parameters file
        with open(self.data_paths[idx]["parameters"]) as file:
            params = file.read()

        return emb, gt, raw / 255.0, params

    # This thing should be efficient as it gets
    def __getitem__(self, idx):
        # get the sample
        sample = self.data_paths[idx]
        # load the GT image
        gt = torchvision.io.read_image(sample["GT"], ImageReadMode.GRAY).float()
        gt = gt.permute(1, 2, 0)  # HWC
        # load embeddings
        integral_images = torch.stack(
            [
                self.processor(
                    torchvision.io.read_image(file, ImageReadMode.GRAY)
                    .float()
                    .repeat(3, 1, 1),
                    return_tensors="pt",
                )["pixel_values"]
                for file in sample["integral_images"][:4]
            ],
            dim=1,
        ).transpose(0, 1)
        # check for nan values
        return integral_images, gt / 255.0  # "normalize" to [0, 1]


class HFImageDataModule(BaseDataModule):
    def __init__(
        self,
            backbone_model_name: str = "facebook/dpt-dinov2-small-nyu",
            data_paths_json_path: str = "../data/data_paths.json",
            data_split: Tuple[float, float, float] = (0.80, 0.1, 0.1),
            batch_size: int = 8,
            num_workers: int = 0,
            pin_memory: bool = False,
            persistent_workers: bool = True,
            data_limit: int = sys.maxsize
    ) -> None:
        """Initialize a DataModule.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param data_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__(data_paths_json_path, data_split, batch_size, num_workers, pin_memory, persistent_workers, data_limit)

        self.backbone_model_name = backbone_model_name


    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        data_paths = []
        with open(self.data_paths_json_path) as file:
            data_paths = json.load(file)

        random.seed(42)
        random.shuffle(data_paths)

        total_items = len(data_paths)
        train_size = int(total_items * self.hparams.data_split[0])
        val_size = int(total_items * self.hparams.data_split[1])

        self.data_train: Optional[Dataset] = HFDataSet(
            data_paths[:train_size],
            self.backbone_model_name,
            self.data_limit
        )
        self.data_val: Optional[Dataset] = HFDataSet(
            data_paths[train_size : train_size + val_size],
            self.backbone_model_name,
            self.data_limit
        )
        self.data_test: Optional[Dataset] = HFDataSet(
            data_paths[train_size + val_size :],
            self.backbone_model_name,
            self.data_limit
        )

        # Divide batch size by the number of devices.
        # Only useful for multiple GPUs, let it be or remove it
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = HFDataSet(self.hparams.data_dir)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset, self.hparams.data_split
            )


if __name__ == "__main__":
    a = HFImageDataModule(data_paths_json_path="data_paths.json")
    a.setup()
    emb, gt, raw, params = a.data_test.get_all(0)
    raw = raw.permute(2, 0, 1)
    # save the raw image
    torchvision.utils.save_image(raw.unsqueeze(0).float(), "raw.png")
    # save the first emb image
    first_emb = emb[1]
    first_emb = (first_emb - first_emb.min()) / (first_emb.max() - first_emb.min())
    torchvision.utils.save_image(first_emb.float(), "emb.png")
    interp_emb = torch.nn.functional.interpolate(
        first_emb, scale_factor=4, mode="nearest"
    )
    interp_emb = interp_emb.squeeze(0)
    torchvision.utils.save_image(interp_emb.float(), "emb_interp.png")
    gt = gt.permute(2, 0, 1)
    # save the gt image
    torchvision.utils.save_image(gt.unsqueeze(0).float(), "gt.png")
    print(emb.shape)
    print(gt.shape)
    print(raw.shape)
    print(params)
