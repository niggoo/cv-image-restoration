import torch

from .image_datamodule import ImageDataModule
from safetensors.torch import load_file
import json
from typing import Optional, Tuple
import random
import torchvision
from torch.utils.data import Dataset, random_split
from torchvision.io import ImageReadMode
from transformers import AutoImageProcessor


class RegularDataSet(Dataset):
    def __init__(
        self, data_paths: dict, model_name: str = "facebook/dpt-dinov2-small-nyu"
    ):
        super().__init__()
        self.data_paths = data_paths
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        # self.processor.crop_size = {"height": 512, "width": 512}

    def __len__(self):
        return len(self.data_paths)

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


class RegularDataModule(ImageDataModule):
    """
    only override the setup method
    """

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

        self.data_train: Optional[Dataset] = RegularDataSet(data_paths[:train_size])
        self.data_val: Optional[Dataset] = RegularDataSet(
            data_paths[train_size : train_size + val_size]
        )
        self.data_test: Optional[Dataset] = RegularDataSet(
            data_paths[train_size + val_size :]
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
            dataset = RegularDataSet(self.hparams.data_dir)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset, self.hparams.data_split
            )


if __name__ == "__main__":
    a = RegularDataModule(data_paths_json_path="data_paths.json")
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