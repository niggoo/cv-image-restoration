import torch
import json
from dinov2_custom import Dinov2ForRestoration
from src.data.dataset_cvproj import CVProjDataset


def plot_images(integral_image, predict, target):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(integral_image)
    ax[0].set_title("Integral Image")
    ax[1].imshow(predict)
    ax[1].set_title("Prediction")
    ax[2].imshow(target)
    ax[2].set_title("Target")
    for a in ax:
        a.set_axis_off()
    plt.show()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the model
    model = Dinov2ForRestoration.from_pretrained("facebook/dinov2-base")
    model.load_state_dict(torch.load("checkpoints/checkpoint_epoch3.pth"))
    model.eval()
    model.to(device)

    # this "test" is with data from the training set
    with open("../data_paths.json") as f:
        data_paths_list = json.load(f)

    data_paths_list = data_paths_list[:500]
    dataset = CVProjDataset(data_paths_list)

    x, y = dataset[3]
    x = x.to(device)
    x = x[:3, :, :]  # only use first 3 channels of integral images because dinov2

    # run the model
    out = model(x.unsqueeze(0))

    # scale images to [0, 255]
    x = x * 255
    y = y * 255
    out = out * 255
    # convert to uint8
    x = x.to(torch.uint8)
    y = y.to(torch.uint8)
    out = out.to(torch.uint8)

    # integral image has more than one channel, so take mean across channels
    x_plot = x.squeeze().cpu().detach().numpy().mean(axis=0)
    plot_images(x_plot, out.squeeze().cpu().detach().numpy(), y.squeeze().detach().numpy())
    print(out)