import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
import numpy as np
from pytorch_msssim import SSIM, MS_SSIM

from typing import Optional

logger = logging.getLogger(__name__)


class WeightedLosses:
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

        # normalized the weights
        self.weights = weights / weights.sum()

        # move weights to gpu is needed
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = self.weights.to(device)

    def get_mse_exp(self, output, target):
        """Loss function weighted MSE """
        return torch.sum(self.weights * (output - target) ** 2)

    def get_mae_exp(self, output, target):
        """Loss function weighted MAE"""
        return torch.sum(self.weights * torch.abs(output - target))


class SSIMLoss(nn.Module):
    def __init__(self, convert_range: bool = False, **kwargs):
        """
        SSIM Loss, optionally converting input range from [-1,1] to [0,1]
        Args:
            convert_range: Convert input from -1,1 to 0,1 range
            **kwargs: Kwargs to pass through to SSIM
        """
        super(SSIMLoss, self).__init__()
        self.convert_range = convert_range
        self.ssim_module = SSIM(**kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.convert_range:
            x = torch.div(torch.add(x, 1), 2)
            y = torch.div(torch.add(y, 1), 2)
        return 1.0 - self.ssim_module(x, y)


class MS_SSIMLoss(nn.Module):
    def __init__(self, convert_range: bool = False, **kwargs):
        """
        Multi-Scale SSIM Loss, optionally converting input range from [-1,1] to [0,1]
        Args:
            convert_range: Convert input from -1,1 to 0,1 range
            **kwargs: Kwargs to pass through to MS_SSIM
        """
        super(MS_SSIMLoss, self).__init__()
        self.convert_range = convert_range
        self.ssim_module = MS_SSIM(**kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.convert_range:
            x = torch.div(torch.add(x, 1), 2)
            y = torch.div(torch.add(y, 1), 2)
        return 1.0 - self.ssim_module(x, y)


class SSIMLossDynamic(nn.Module):
    def __init__(self, convert_range: bool = False, **kwargs):
        """
        SSIM Loss on only dynamic part of the images, optionally converting input range from [-1,1] to [0,1]

        In Mathieu et al. to stop SSIM regressing towards the mean and predicting only the background, they only
        run SSIM on the dynamic parts of the image. We can accomplish that by subtracting the current image from the future ones

        Args:
            convert_range: Whether to convert from -1,1 to 0,1 as required for SSIM
            **kwargs: Kwargs for the ssim_module
        """
        super(SSIMLossDynamic, self).__init__()
        self.convert_range = convert_range
        self.ssim_module = MS_SSIM(**kwargs)

    def forward(self, current_image: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
        """
        Args:
            current_image: The last 'real' image given to the mode
            x: The target future sequence
            y: The predicted future sequence

        Returns:
            The SSIM loss computed only for the parts of the image that has changed
        """
        if self.convert_range:
            current_image = torch.div(torch.add(current_image, 1), 2)
            x = torch.div(torch.add(x, 1), 2)
            y = torch.div(torch.add(y, 1), 2)
        # Subtract 'now' image to get what changes for both x and y
        x = x - current_image
        y = y - current_image
        # TODO: Mask out loss from pixels that don't change
        return 1.0 - self.ssim_module(x, y)


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


class GradientDifferenceLoss(nn.Module):
    """
    Gradient Difference Loss that penalizes blurry images more than MSE
    """

    def __init__(self, alpha: int = 2):
        super(GradientDifferenceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor):
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
        t1 = F.pad(input=t1, pad=(0, 0, 1, 0), mode='constant', value=0)
        t2 = F.pad(input=t2, pad=(0, 1, 0, 0), mode='constant', value=0)
        loss = t1 + t2
        return loss.mean()


class GridCellLoss(nn.Module):
    """
    Grid Cell Regularizer loss from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

    """

    def __init__(self, weight_fn=None):
        super().__init__()
        self.weight_fn = weight_fn  # In Paper, weight_fn is max(y+1,24)

    def forward(self, generated_images, targets):
        """
        Calculates the grid cell regularizer value, assumes generated images are the mean predictions from
        6 calls to the generater (Monte Carlo estimation of the expectations for the latent variable)
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
    Loss described in Skillful-Nowcasting GAN,  see https://arxiv.org/pdf/2104.00954.pdf
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, real_flag):
        if real_flag is True:
            x = -x
        return F.relu(1.0 + x).mean()


class FocalLoss(nn.Module):
    def __init__(
        self,
        apply_nonlin=None,
        alpha=None,
        gamma=2,
        balance_index=0,
        smooth=1e-5,
        size_average=True,
    ):
        """
        Copied from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py

        This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
        'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)

        Args:
            apply_nonlin: nn.Module, a non-linearity to apply to the logits
            alpha: (tensor) 3D or 4D the scalar factor for this criterion
            gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
            balance_index: balance class index, should be specific when alpha is float
            smooth: (float,double) smooth value when cross entropy
            size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
        """
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError("Not support alpha type")

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth
            )
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def get_loss(loss: str = "mse", **kwargs) -> torch.nn.Module:
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
        criterion = TotalVariationLoss(tv_weight=kwargs.get("tv_weight", 1))
    elif loss in ["gdl", "gradient_difference_loss"]:
        criterion = GradientDifferenceLoss(alpha=kwargs.get("alpha", 2))
    elif loss in ["weighted_mse", "weighted_mae"]:
        criterion = WeightedLosses(**kwargs)
    else:
        raise ValueError(f"loss {loss} not recognized")
    return criterion
