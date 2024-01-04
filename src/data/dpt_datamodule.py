import os.path
import sys

import torch
import numpy as np

from src.data.base_datamodule import BaseDataModule
import json
from typing import Optional, Tuple
import random
import torchvision
from torch.utils.data import Dataset, random_split
from torchvision.io import ImageReadMode
from tqdm import tqdm


class DptDataSet(Dataset):
    def __init__(self, data_paths: dict,
                 norm_stat: torch.Tensor,
                 data_limit: int = sys.maxsize,):
        super().__init__()
        self.data_paths = data_paths
        self.data_limit = data_limit
        self.norm = torchvision.transforms.Normalize(mean= norm_stat[:, 0], std=norm_stat[:, 1])

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
        gt = torchvision.io.read_image(sample["GT"], ImageReadMode.GRAY)
        # load integrals
        integral_images = self.norm(torch.stack(
            [
                torchvision.io.read_image(file, ImageReadMode.GRAY).float().squeeze()
                for file in sample["integral_images"][:4]
            ],
            dim=0,
        ))
        # check for nan values
        return integral_images, gt / 255.0  # "normalize" to [0, 1]


def get_norm(sample: dict):
    stats = torch.empty(4, 2)
    for i, f in enumerate(sample["integral_images"][:4]):
        img = torchvision.io.read_image(f, ImageReadMode.GRAY).float()
        stats[i, 0] = torch.mean(img)
        stats[i, 1] = torch.std(img)
    return stats


def calc_norm(data_paths: list):

    lstats = []
    with torch.multiprocessing.Pool(6) as p:
        for s in tqdm(p.imap_unordered(get_norm, data_paths)):
            lstats.append(s)

    lstats = torch.stack(lstats, dim=0)
    return lstats.mean(dim=0)


class DptImageDataModule(BaseDataModule):
    def __init__(
        self,
            data_paths_json_path: str = "../data/data_paths.json",
            data_split: Tuple[float, float, float] = (0.80, 0.1, 0.1),
            batch_size: int = 8,
            num_workers: int = 0,
            pin_memory: bool = False,
            persistent_workers: bool = True,
            oversample: bool = False,
            data_limit: int = sys.maxsize
    ) -> None:
        """Initialize a DataModule.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param data_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__(data_paths_json_path, data_split, batch_size, num_workers, pin_memory, persistent_workers, oversample, data_limit)

        self.oversample = oversample

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

        ns_path = os.path.dirname(self.data_paths_json_path)
        ns_path = os.path.join(ns_path, "norm_stats.pt")
        if os.path.exists(ns_path):
            norm_stats = torch.load(ns_path)
        else:
            print("Calculate Norm for train set:")
            norm_stats = calc_norm(data_paths[:train_size])
            torch.save(norm_stats, ns_path)

        self.data_train: Optional[Dataset] = DptDataSet(
            data_paths[:train_size],
            norm_stats,
            self.data_limit
        )
        self.data_val: Optional[Dataset] = DptDataSet(
            data_paths[train_size : train_size + val_size],
            norm_stats,
            self.data_limit
        )
        self.data_test: Optional[Dataset] = DptDataSet(
            data_paths[train_size + val_size:],
            norm_stats,
            self.data_limit
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
    a = DptImageDataModule(data_paths_json_path="data_paths.json")
    a.setup()
    emb, gt, raw, params = a.data_test.get_all(0)
    raw = raw.permute(2, 0, 1)
    # save the raw image
    torchvision.utils.save_image(raw.unsqueeze(0).float(), "raw.png")
    # save the first emb image
    first_emb = emb[0].unsqueeze(0)
    first_emb = (first_emb - first_emb.min()) / (first_emb.max() - first_emb.min())
    torchvision.utils.save_image(first_emb.float(), "emb.png")
    interp_emb = torch.nn.functional.interpolate(
        first_emb, scale_factor=4, mode="nearest"
    )
    interp_emb = interp_emb.squeeze(0)
    torchvision.utils.save_image(interp_emb.float(), "emb_interp.png")

    # save the gt image
    torchvision.utils.save_image(gt.unsqueeze(0).float(), "gt.png")
    print(emb.shape)
    print(gt.shape)
    print(raw.shape)
    print(params)
