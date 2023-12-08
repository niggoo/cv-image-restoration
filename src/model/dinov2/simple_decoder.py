import torch


class SimpleDecoder(torch.nn.Module):
    def __init__(self, in_channels=384, tokenW=36, tokenH=36, num_labels=1, image_size=(512, 512)):
        super(SimpleDecoder, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.image_size = image_size
        # four convs for the four focal planes
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=1, stride=1)
        self.conv3 = torch.nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=1, stride=1)
        self.conv4 = torch.nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=1, stride=1)

        # final conv that combines the four focal planes
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1),
            torch.nn.Sigmoid()
        )

    def forward(self, embeddings):
        batch_size, focal_planes, _, hidden_size = embeddings.shape

        embeddings = embeddings.reshape(batch_size, focal_planes, self.height, self.width, hidden_size)
        embeddings = embeddings.permute(0, 1, 4, 2, 3)  # from height, width, hidden_size to hidden_size, height, width

        # this might be the same as one conv and the focal planes stacked along the hidden_size (channel)
        embeddings1 = self.conv1(embeddings[:, 0, :, :, :])
        embeddings2 = self.conv2(embeddings[:, 1, :, :, :])
        embeddings3 = self.conv3(embeddings[:, 2, :, :, :])
        embeddings4 = self.conv4(embeddings[:, 3, :, :, :])

        # concatenate the four focal planes
        embeddings = torch.cat((embeddings1, embeddings2, embeddings3, embeddings4), dim=1)
        embeddings = torch.nn.functional.relu(embeddings)

        pred = self.classifier(embeddings)
        # upsample to the original image size
        pred = torch.nn.functional.interpolate(pred, size=self.image_size, mode="bilinear",
                                               align_corners=False)

        return pred.permute(0, 2, 3, 1)  # NCHW -> NHWC


if __name__ == "__main__":
    model = SimpleDecoder(384, 36, 36, 1, (512, 512))
    pred = model(torch.randn(8, 4, 36*36, 384))
    print(pred.shape)
