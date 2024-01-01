import json
import sys
from typing import Optional, Tuple
import random
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode

from src.data.base_datamodule import BaseDataModule

# For debugging poses
#from base_datamodule import BaseDataModule
#import re

class ImageDataSet(Dataset):
    def __init__(self, data_paths: dict, mean, std, data_limit:int = sys.maxsize):
        super().__init__()
        self.data_paths = data_paths
        self.mean = mean
        self.std = std
        self.data_limit = data_limit

        #DEBUG poses
        #self.pose_reg = re.compile(r"person shape =  (.*)\n")

    def __len__(self):
        return min(len(self.data_paths), self.data_limit)

    def get_all(self, idx):
        emb, gt = self.__getitem__(idx)
        # load the raw image
        raw = torchvision.io.read_image(
            self.data_paths[idx]["raw_images"][0], ImageReadMode.GRAY
        ).float()
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
        # load integral images and stack along the channel dimension
        integral_images = (
            torch.stack(
                [
                    torchvision.io.read_image(image, ImageReadMode.GRAY)
                    for image in sample["integral_images"][:4]
                ],
                dim=0,
            )
            .squeeze()
            .float()
        )

        return (
            (integral_images - self.mean) / self.std
        ), gt / 255.0  # "normalize" to [0, 1]
    
        #DEBUG poses
        # with open(sample["parameters"]) as file:
        #     params = file.read()
        
        # if "person shape" in params:
        #     pose = self.pose_reg.search(params).group(1)
        # else:
        #     pose = "no person"
        
        # return pose


class ImageDataModule(BaseDataModule):
    def __init__(
        self,
        mean: float,
        std: float,
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
        super().__init__(
            data_paths_json_path,
            data_split,
            batch_size,
            num_workers,
            pin_memory,
            persistent_workers,
            data_limit
        )

        self.mean = mean
        self.std = std

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

        self.data_train: Optional[Dataset] = ImageDataSet(
            data_paths[:train_size], self.mean, self.std, self.data_limit
        )
        self.data_val: Optional[Dataset] = ImageDataSet(
            data_paths[train_size : train_size + val_size], self.mean, self.std, self.data_limit
        )
        self.data_test: Optional[Dataset] = ImageDataSet(
            data_paths[train_size + val_size :], self.mean, self.std, self.data_limit
        )

        self.data_paths = data_paths

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


if __name__ == "__main__":
    _ = ImageDataModule()
