# source: https://github.com/milesial/Pytorch-UNet
# source: https://pub.towardsai.net/dinov2-for-custom-dataset-segmentation-a-comprehensive-tutorial-1cd749b3eeda
# # https://colab.research.google.com/drive/1UMQj7F_x0fSy_gevlTZ9zLYn7b02kTqi?usp=sharing#scrollTo=rLzR_mt_SnE2

import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import json

from dataset_cvproj import CVProjDataset
from dinov2_custom import Dinov2ForRestoration

dir_checkpoint = Path('checkpoints/')

with open("../data_paths.json") as f:
    data_paths_list = json.load(f)

# !!! only take some of the data for testing purposes
data_paths_list = data_paths_list[:5]


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 3,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    dataset = CVProjDataset(data_paths_list)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='Custom_dinov2', resume='allow', anonymous='must', entity="pauldd")
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss(reduction='mean')
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                integral_images, target_image = batch
                # only use first 3 channels of integral images because dinov2
                integral_images = integral_images[:, :3, :, :]

                integral_images = integral_images.to(device=device, dtype=torch.float32,
                                                     memory_format=torch.channels_last)
                target_image = target_image.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    prediction = model(integral_images)
                    loss = criterion(prediction, target_image)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(integral_images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                if global_step % 4500 == 0:
                    # evaluate on validation set
                    val_score = 0
                    model.eval()
                    with torch.no_grad():
                        for val_batch in val_loader:
                            integral_images, target_image = val_batch
                            integral_images = integral_images[:, :3, :, :]
                            integral_images = integral_images.to(device=device, dtype=torch.float32,
                                                                 memory_format=torch.channels_last)
                            target_image = target_image.to(device=device, dtype=torch.float32)
                            prediction = model(integral_images)
                            val_score += criterion(prediction, target_image).item()
                    val_score /= len(val_loader)
                    model.train()
                    scheduler.step(val_score)

                    logging.info('Validation score: {}'.format(val_score))
                    try:
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation score': val_score,
                            'images': wandb.Image(integral_images[0].cpu()),
                            'output': {
                                'true': wandb.Image(target_image[0].float().cpu()),
                                'pred': wandb.Image(prediction.argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                        })
                    except:
                        pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and targets')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--in_channels', '-c', type=int, default=4, help='Number of input channels')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = Dinov2ForRestoration.from_pretrained("facebook/dinov2-base")

    for name, param in model.named_parameters():
        if name.startswith("dinov2"):
            param.requires_grad = False

    logging.info(f'Network:\n'
                 f'\t{model.in_channels} input channels\n'
                 f'\t{model.out_channels} output channels\n')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)

    train_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_scale=args.scale,
        val_percent=args.val / 100,
        amp=args.amp
    )
