import torch
import torchvision
from torchvision.io.image import ImageReadMode
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class CVProjDataset(Dataset):
    """
    CVProj dataset
    Uses the data_paths_list generated by generate_data_json.py
    data_paths_list: list of dicts with keys "batch", "image_id", "GT", "raw_images", "parameters", "integral_images"
    transform_x: transform to apply to the integral images
    transform_y: transform to apply to the GT image
    focal_planes: list of focal planes to use, must have been generated before

    """

    def __init__(self, data_paths, transform_x=None, transform_y=None):
        self.data_paths = data_paths
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # get the sample
        sample = self.data_paths[idx]
        # load the GT image
        gt = torchvision.io.read_image(sample["GT"], ImageReadMode.GRAY).float()
        # load integral images and stack along the channel dimension
        integral_images = torch.stack([torchvision.io.read_image(image, ImageReadMode.GRAY)
                                       for image in sample["integral_images"]], dim=0).squeeze().float()

        # apply the transform
        if self.transform_x:
            integral_images = self.transform_x(integral_images)
        if self.transform_y:
            gt = self.transform_y(gt)

        return integral_images / 255.0, gt / 255.0  # "normalize" to [0, 1]


if __name__ == "__main__":
    # load the data_paths_list
    import json

    with open("data_paths.json") as f:
        data_paths_list = json.load(f)

    # create the dataset
    dataset = CVProjDataset(data_paths_list)

    # print dataset stats
    print("Number of samples: ", len(dataset))

    # create a dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in dataloader:
        integral_images, gt = batch
        print("Integral images shape: ", integral_images.shape)
        print("GT shape: ", gt.shape)
        break
