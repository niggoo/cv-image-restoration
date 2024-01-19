"""
https://github.com/isl-org/DPT/blob/main/EVALUATION.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSGE(nn.Module):
    """
    Mean gradient squared error
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels
        self.kernel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.kernel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

    def forward(self, pred, target):
        # Ensure the input tensors are on the same device as the model
        self.kernel_x = self.kernel_x.to(pred.device)
        self.kernel_y = self.kernel_y.to(pred.device)

        # bring pred and target to shape (N, 1, H, W)
        # pred = pred.permute(0, 3, 1, 2)
        # target = target.permute(0, 3, 1, 2)

        # Calculate gradients for input and target
        grad_input_x = F.conv2d(pred, self.kernel_x, padding=1)
        grad_input_y = F.conv2d(pred, self.kernel_y, padding=1)
        grad_target_x = F.conv2d(target, self.kernel_x, padding=1)
        grad_target_y = F.conv2d(target, self.kernel_y, padding=1)

        # Calculate Mean Squared Gradient Error
        loss_x = F.mse_loss(grad_input_x, grad_target_x)
        loss_y = F.mse_loss(grad_input_y, grad_target_y)

        return loss_x + loss_y  # / 2
