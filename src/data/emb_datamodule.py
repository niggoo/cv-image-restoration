import sys

import torch

from src.data.base_datamodule import BaseDataModule
from safetensors.torch import load_file
import json
from typing import Optional
import random
import torchvision
from torch.utils.data import Dataset, random_split
from torchvision.io import ImageReadMode


class EmbeddingDataSet(Dataset):
    def __init__(self, data_paths: dict, data_limit: int = sys.maxsize):
        super().__init__()
        self.data_paths = data_paths
        self.data_limit = data_limit

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
        embeddings = torch.stack(
            [
                load_file(file)["last_hidden_states"]
                for file in sample["embeddings"][:4]
            ],
            dim=1,
        ).squeeze(0)
        embeddings = embeddings[:, 1:, :].float()  # remove CLS token

        return embeddings, gt / 255.0  # "normalize" to [0, 1]


class EmbeddingDataModule(BaseDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        with open(self.data_paths_json_path) as file:
            data_paths = json.load(file)

        random.seed(42)
        random.shuffle(data_paths)

        total_items = len(data_paths)
        train_size = int(total_items * self.hparams.data_split[0])
        val_size = int(total_items * self.hparams.data_split[1])

        self.data_train: Optional[Dataset] = EmbeddingDataSet(
            data_paths[:train_size],
            self.data_limit
        )
        self.data_val: Optional[Dataset] = EmbeddingDataSet(
            data_paths[train_size : train_size + val_size],
            self.data_limit
        )
        self.data_test: Optional[Dataset] = EmbeddingDataSet(
            data_paths[train_size + val_size :],
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

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = EmbeddingDataSet(self.hparams.data_dir)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset, self.hparams.data_split
            )


if __name__ == "__main__":
    a = EmbeddingDataModule(data_paths_json_path="data_paths.json")
    a.setup()
    emb, gt, raw, params = a.data_test.get_all(0)
    print(emb.shape)
    print(gt.shape)
    print(raw.shape)
    print(params)
