import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

class ModifiedConvHead(nn.Module):
    """Expanded convolutional head for the DINOv2 model."""
    def __init__(self, in_channels=384, tokenW=36, tokenH=36, num_labels=1, image_size=(512, 512)):
        super(ModifiedConvHead, self).__init__()
        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.image_size = image_size

        # Individual convolutions for each focal plane
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, 8, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels, 8, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels, 8, kernel_size=1, stride=1)

        # Final convolution that combines the outputs
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 64, (1, 1), stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, num_labels, (3, 3), padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, embeddings):
        # Reshape and rearrange embeddings for convolution
        embeddings = rearrange(embeddings, 'b f (h w) c -> b f c h w', h=self.height, w=self.width)

        # Apply convolutions to each focal plane
        conv1_out = self.conv1(embeddings[:, 0, :, :, :])
        conv2_out = self.conv2(embeddings[:, 1, :, :, :])
        conv3_out = self.conv3(embeddings[:, 2, :, :, :])
        conv4_out = self.conv4(embeddings[:, 3, :, :, :])
        
        # Concatenate the outputs of the convolutions
        combined = torch.cat([conv1_out, conv2_out, conv3_out, conv4_out], dim=1)

        # Apply the final convolution
        output = self.final_conv(combined)

        # Upsample to the original image size
        output = F.interpolate(output, size=self.image_size, mode='bilinear', align_corners=False)

        return rearrange(output, 'b c h w -> b h w c')  # NCHW -> NHWC


if __name__ == "__main__":
    model = ModifiedConvHead(768, 36, 36, 1, (512, 512))
    pred = model(torch.randn(8, 4, 36 * 36, 768))
    print(pred.shape)
