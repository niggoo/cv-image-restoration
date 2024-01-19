import torch
import torch.nn as nn
import torchinfo
from transformers import Dinov2Model, Dinov2Config


class Dinov2(nn.Module):
    def __init__(
        self,
        dinov2_size: str = "small",
        out_features: list = [3, 6, 9, 12],
        freeze_encoder: bool = True,
        channels: int = 4,
        skip: bool = False,
    ) -> None:
        """Dinov2 model with a selection of output features

        :params dinov2_size: size of the model, either "small", "base", "large" or "giant"
        """
        super(Dinov2, self).__init__()
        model_name = f"facebook/dinov2-{dinov2_size}"
        config = Dinov2Config.from_pretrained(model_name)
        # overwrite settings
        config.num_channels = channels  # input channels
        config.patch_size = 16
        config.output_hidden_states = True
        self.out_features = out_features
        self.config = config
        # init dinov2 model
        self.model = Dinov2Model.from_pretrained(
            model_name, config=config, ignore_mismatched_sizes=True
        )
        # freeze encoder but leave embeddings trainable
        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        # skip connection
        if skip:
            self.skip = nn.Conv2d(
                in_channels=channels,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        else:
            self.skip = None

    def forward(self, x):
        """
        Takes a picture in the form (B, IN_C, W, H) and
        returns a stack of features in the form of ((B, H, W/P, H/P), CLS)
        :param x: batch of pictures
        :return: stack of features
        """
        # save shape for later
        B, _, w, h = x.shape
        # skip connection
        s = self.skip(x) if self.skip is not None else None
        # get features and select the relevant stages
        x = self.model(x)
        # reformat output
        outputs = list(x.hidden_states[i] for i in self.out_features)
        outputs = [
            (out[:, 1 + 0 :], out[:, 0]) for out in outputs
        ]  # zero is the number of register (should be always 0)
        outputs = [
            (
                seq.reshape(
                    B, w // self.config.patch_size, h // self.config.patch_size, -1
                )
                .permute(0, 3, 1, 2)
                .contiguous(),
                cls,
            )
            for seq, cls in outputs
        ]
        return outputs, s


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = Dinov2(dinov2_size="small").to(device)
    x = torch.rand((8, 4, 512, 512)).to(device)
    torchinfo.summary(module, x.shape)
    x = module(x)
    for i in x:
        print(i[0].shape, i[1].shape)
