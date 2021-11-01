"""
This file contains various versions of losses using the Structural Similarity Index Measure.

See https://en.wikipedia.org/wiki/Structural_similarity for more details

"""

import torch
from pytorch_msssim import MS_SSIM, SSIM
from torch import nn as nn


class SSIMLoss(nn.Module):
    """SSIM Loss, optionally converting input range from [-1,1] to [0,1]"""

    def __init__(self, convert_range: bool = False, **kwargs):
        """
        Init

        Args:
            convert_range: Convert input from -1,1 to 0,1 range
            **kwargs: Kwargs to pass through to SSIM
        """
        super(SSIMLoss, self).__init__()
        self.convert_range = convert_range
        self.ssim_module = SSIM(**kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Forward method

        Args:
            x: one tensor
            y: second tensor

        Returns: SSIM loss

        """
        if self.convert_range:
            x = torch.div(torch.add(x, 1), 2)
            y = torch.div(torch.add(y, 1), 2)
        return 1.0 - self.ssim_module(x, y)


class MS_SSIMLoss(nn.Module):
    """Multi-Scale SSIM Loss, optionally converting input range from [-1,1] to [0,1]"""

    def __init__(self, convert_range: bool = False, **kwargs):
        """
        Initialize

        Args:
            convert_range: Convert input from -1,1 to 0,1 range
            **kwargs: Kwargs to pass through to MS_SSIM
        """
        super(MS_SSIMLoss, self).__init__()
        self.convert_range = convert_range
        self.ssim_module = MS_SSIM(**kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Forward method

        Args:
            x: tensor one
            y: tensor two

        Returns:M S SSIM Loss

        """
        if self.convert_range:
            x = torch.div(torch.add(x, 1), 2)
            y = torch.div(torch.add(y, 1), 2)
        return 1.0 - self.ssim_module(x, y)


class SSIMLossDynamic(nn.Module):
    """
    SSIM Loss on only dynamic part of the images

    Optionally converting input range from [-1,1] to [0,1]

    In Mathieu et al. to stop SSIM regressing towards the mean and predicting
    only the background, they only run SSIM on the dynamic parts of the image.
    We can accomplish that by subtracting the current image from the future ones
    """

    def __init__(self, convert_range: bool = False, **kwargs):
        """
        Initialize

        Args:
            convert_range: Whether to convert from -1,1 to 0,1 as required for SSIM
            **kwargs: Kwargs for the ssim_module
        """
        super(SSIMLossDynamic, self).__init__()
        self.convert_range = convert_range
        self.ssim_module = MS_SSIM(**kwargs)

    def forward(self, current_image: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
        """
        Forward method

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
