# source: https://pub.towardsai.net/dinov2-for-custom-dataset-segmentation-a-comprehensive-tutorial-1cd749b3eeda
# https://colab.research.google.com/drive/1UMQj7F_x0fSy_gevlTZ9zLYn7b02kTqi?usp=sharing#scrollTo=rLzR_mt_SnE2

import torch
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=36, tokenH=36, out_channels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        return self.classifier(embeddings)


class Dinov2ForRestoration(Dinov2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.in_channels = config.hidden_size
        self.out_channels = 3

        self.dinov2 = Dinov2Model(config)
        self.classifier = LinearClassifier(self.in_channels, 36, 36, self.out_channels)

        self.combine = torch.nn.Conv2d(in_channels=self.out_channels + 3, out_channels=1, kernel_size=1)

    def forward(self, pixel_values, output_hidden_states=False, output_attentions=False):
        # use frozen features
        outputs = self.dinov2(pixel_values,
                              output_hidden_states=output_hidden_states,
                              output_attentions=output_attentions)
        # dinov2 divides image into 14 patches
        # input of 3x512x512 images --> 512//14 = 36, square patches --> 36x36 = 1296
        # hidden size = 768
        # output shape = (batch_size, number of patches, hidden size) = (2, 1296, 768)

        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]

        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)
        logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear",
                                                 align_corners=False)
        # combine with original image
        combined = self.combine(torch.cat((logits, pixel_values), dim=1))
        res = torch.sigmoid(combined)
        return res


if __name__ == "__main__":
    model = Dinov2ForRestoration.from_pretrained("facebook/dinov2-base")
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using {device}")
    # create a random input
    x = torch.rand((2, 3, 512, 512)).to(device)

    # run the model
    out = model(x)
    print(out.shape)  # should be [2, 1, 512, 512]
