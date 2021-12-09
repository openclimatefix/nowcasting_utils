"""Sereval loss functions and high level loss function get'er."""
import logging
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from nowcasting_utils.models.losses.FocalLoss import FocalLoss
from nowcasting_utils.models.losses.StructuralSimilarity import (
    MS_SSIMLoss,
    SSIMLoss,
    SSIMLossDynamic,
)
from nowcasting_utils.models.losses.TotalVariationLoss import TVLoss

logger = logging.getLogger(__name__)


class WeightedLosses:
    """Class: Weighted loss depending on the forecast horizon."""

    def __init__(self, decay_rate: Optional[int] = None, forecast_length: int = 6):
        """
        Want to set up the MSE loss function so the weights only have to be calculated once.

        Args:
            decay_rate: The weights exponentially decay depending on the 'decay_rate'.
            forecast_length: The forecast length is needed to make sure the weights sum to 1
        """
        self.decay_rate = decay_rate
        self.forecast_length = forecast_length

        logger.debug(
            f"Setting up weights with decay rate {decay_rate} and of length {forecast_length}"
        )

        # set default rate of ln(2) if not set
        if self.decay_rate is None:
            self.decay_rate = math.log(2)

        # make weights from decay rate
        weights = torch.FloatTensor(
            [math.exp(-self.decay_rate * i) for i in range(0, self.forecast_length)]
        )

        # normalized the weights, so there mean is 1.
        # To calculate the loss, we times the weights by the differences between truth
        # and predictions and then take the mean across all forecast horizons and the batch
        self.weights = weights / weights.sum() * len(weights)

        # move weights to gpu is needed
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = self.weights.to(device)

    def get_mse_exp(self, output, target):
        """Loss function weighted MSE"""

        # get the differences weighted by the forecast horizon weights
        diff_with_weights = self.weights * ((output - target) ** 2)

        # average across batches
        return torch.mean(diff_with_weights)

    def get_mae_exp(self, output, target):
        """Loss function weighted MAE"""

        # get the differences weighted by the forecast horizon weights
        diff_with_weights = self.weights * torch.abs(output - target)

        # average across batches
        return torch.mean(diff_with_weights)


class GradientDifferenceLoss(nn.Module):
    """
    Gradient Difference Loss that penalizes blurry images more than MSE.
    """

    def __init__(self, alpha: int = 2):
        """
        Initalize the Loss Class.

        Args:
            alpha: #TODO
        """
        super(GradientDifferenceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Calculate the Gradient Difference Loss.

        Args:
            x: vector one
            y: vector two

        Returns: the Gradient Difference Loss value

        """
        t1 = torch.pow(
            torch.abs(
                torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
                - torch.abs(y[:, :, :, 1:, :] - y[:, :, :, :-1, :])
            ),
            self.alpha,
        )
        t2 = torch.pow(
            torch.abs(
                torch.abs(x[:, :, :, :, :-1] - x[:, :, :, :, 1:])
                - torch.abs(y[:, :, :, :, :-1] - y[:, :, :, :, 1:])
            ),
            self.alpha,
        )
        # Pad out the last dim in each direction so shapes match
        t1 = F.pad(input=t1, pad=(0, 0, 1, 0), mode="constant", value=0)
        t2 = F.pad(input=t2, pad=(0, 1, 0, 0), mode="constant", value=0)
        loss = t1 + t2
        return loss.mean()


class GridCellLoss(nn.Module):
    """
    Grid Cell Regularizer loss from Skillful Nowcasting,

    see https://arxiv.org/pdf/2104.00954.pdf.
    """

    def __init__(self, weight_fn=None):
        """
        Initialize the model.

        Args:
            weight_fn: the weight function the be called when #TODO?
        """
        super().__init__()
        self.weight_fn = weight_fn  # In Paper, weight_fn is max(y+1,24)

    def forward(self, generated_images, targets):
        """
        Calculates the grid cell regularizer value.

        This assumes generated images are the mean predictions from
        6 calls to the generater
        (Monte Carlo estimation of the expectations for the latent variable)

        Args:
            generated_images: Mean generated images from the generator
            targets: Ground truth future frames

        Returns:
            Grid Cell Regularizer term
        """
        difference = generated_images - targets
        if self.weight_fn is not None:
            difference *= self.weight_fn(targets)
        difference /= targets.size(1) * targets.size(3) * targets.size(4)  # 1/HWN
        return difference.mean()


class NowcastingLoss(nn.Module):
    """
    Loss described in Skillful-Nowcasting GAN,  see https://arxiv.org/pdf/2104.00954.pdf.
    """

    def __init__(self):
        """Initialize the model."""
        super().__init__()

    def forward(self, x, real_flag):
        """
        Forward step.

        Args:
            x: the data to work with
            real_flag: boolean if its real or not

        Returns: #TODO

        """
        if real_flag is True:
            x = -x
        return F.relu(1.0 + x).mean()


def get_loss(loss: str = "mse", **kwargs) -> torch.nn.Module:
    """
    Function to get different losses easily.

    Args:
        loss: name of the loss, or torch.nn.Module, if a Module, returns that Module
        **kwargs: kwargs to pass to the loss function

    Returns:
        torch.nn.Module
    """
    if isinstance(loss, torch.nn.Module):
        return loss
    assert loss in [
        "mse",
        "bce",
        "binary_crossentropy",
        "crossentropy",
        "focal",
        "ssim",
        "ms_ssim",
        "l1",
        "tv",
        "total_variation",
        "ssim_dynamic",
        "gdl",
        "gradient_difference_loss",
        "weighted_mse",
        "weighted_mae",
    ]
    if loss == "mse":
        criterion = F.mse_loss
    elif loss in ["bce", "binary_crossentropy", "crossentropy"]:
        criterion = F.nll_loss
    elif loss in ["focal"]:
        criterion = FocalLoss()
    elif loss in ["ssim"]:
        criterion = SSIMLoss(data_range=1.0, size_average=True, **kwargs)
    elif loss in ["ms_ssim"]:
        criterion = MS_SSIMLoss(data_range=1.0, size_average=True, **kwargs)
    elif loss in ["ssim_dynamic"]:
        criterion = SSIMLossDynamic(data_range=1.0, size_average=True, **kwargs)
    elif loss in ["l1"]:
        criterion = torch.nn.L1Loss()
    elif loss in ["tv", "total_variation"]:
        criterion = TVLoss(
            tv_type=kwargs.get("tv_type", "tv"),
            p=kwargs.get("p", 1),
            reduction=kwargs.get("reduction", "mean"),
        )
    elif loss in ["gdl", "gradient_difference_loss"]:
        criterion = GradientDifferenceLoss(alpha=kwargs.get("alpha", 2))
    elif loss in ["weighted_mse", "weighted_mae"]:
        criterion = WeightedLosses(**kwargs)
    else:
        raise ValueError(f"loss {loss} not recognized")
    return criterion
