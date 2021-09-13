import torch
from torch import nn as nn


def tv_loss(img, tv_weight):
    """
    Taken from https://github.com/chongyangma/cs231n/blob/master/assignments/assignment3/style_transfer_pytorch.py
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, C, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss


class TotalVariationLoss(nn.Module):
    def __init__(self, tv_weight: float = 1.0):
        super(TotalVariationLoss, self).__init__()
        self.tv_weight = tv_weight

    def forward(self, x: torch.Tensor):
        return tv_loss(x, self.tv_weight)