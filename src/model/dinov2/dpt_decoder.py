import sys

sys.path.append("../..")

from transformers import (
    Dinov2Config,
    DPTConfig,
    DPTForDepthEstimation,
    DPTFeatureExtractor,
)
import torch
from einops import rearrange
from torchinfo import summary
import torchvision

# from data.regular_datamodule import RegularDataModule as DataModule

from transformers.models.dpt.modeling_dpt import DPTNeck, DPTDepthEstimationHead


class DPTForReconstruction(torch.nn.Module):
    def __init__(self, model_name: str = "facebook/dpt-dinov2-small-nyu"):
        super(DPTForReconstruction, self).__init__()
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        self.model.config.neck_hidden_states = [64, 128, 256, 512]

        # freeze encoder weights
        for n, p in self.model.named_parameters():
            if "encoder" in n:
                p.requires_grad = False

        self.model.neck = DPTNeck(self.model.config)
        self.model.head = DPTDepthEstimationHead(self.model.config)
        self.conv1d = torch.nn.Conv2d(4, 1, 1)

    def forward(self, x):
        x = rearrange(x, "b f 1 c h w -> (b f) c h w")
        # scale down images to half size
        x = torch.nn.functional.interpolate(x, scale_factor=0.25, mode="bicubic")
        x = self.model(x).predicted_depth
        # take max over the focal planes
        # x = x.max(dim=1)[0]
        x = rearrange(x, "(b f) h w -> b f h w", f=4)
        x = self.conv1d(x)
        # scale up to original size for (h, w) in (b, h, w)
        # 1 dimension is needed for the interpolation to work
        # x = rearrange(x, "b h w -> b 1 h w")
        x = torch.nn.functional.interpolate(x, size=(512, 512), mode="bicubic")
        x = rearrange(x, "b 1 h w -> b h w 1")
        return torch.sigmoid(x)


if __name__ == "__main__":
    # a = DataModule(data_paths_json_path="../../data/data_paths.json")
    # a.setup()
    # model = DPTForReconstruction()
    # print(summary(model, depth=2))
    # # inputs: in range [0, 1]
    # emb, gt, raw, params = a.data_test.get_all(2)
    # # add batch dimension
    # first_emb = emb[1]
    # first_emb = (first_emb - first_emb.min()) / (first_emb.max() - first_emb.min())
    # torchvision.utils.save_image(first_emb.float(), "emb0.png")
    # emb = emb.unsqueeze(0)
    # pred = model(emb)
    # print(pred.shape)
    # pred = (pred - pred.min()) / (pred.max() - pred.min())
    # torchvision.utils.save_image(pred.squeeze(-1), "test.png")
    # torchvision.utils.save_image(gt.permute(2, 0, 1).unsqueeze(0).float(), "gt.png")
    # torchvision.utils.save_image(raw.permute(2, 0, 1).unsqueeze(0).float(), "raw.png")
    pass
