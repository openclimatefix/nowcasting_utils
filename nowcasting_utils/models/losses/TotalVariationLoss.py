"""
Implementation of Total Variation Loss

(https://en.wikipedia.org/wiki/Total_variation_denoising) copied and slightly
modified from the original Apache License 2.0 traiNNer
Authors https://github.com/victorca25/traiNNer/tree/master

# Copyright 2021 traiNNer Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

import torch
from torch import nn as nn
from torch.nn import functional as F


def get_outnorm(x: torch.Tensor, out_norm: str = "") -> torch.Tensor:
    """
    Common function to get a loss normalization value.

    Can normalize by either the
    - batch size ('b'),
    - the number of channels ('c'),
    - the image size ('i')
    - or combinations ('bi', 'bci', etc)

    Args:
        x: the tensor to be normalized
        out_norm: the string dimension to be normalized

    Returns: the normalized tensor

    """
    # b, c, h, w = x.size()
    img_shape = x.shape

    if not out_norm:
        return 1

    norm = 1
    if "b" in out_norm:
        # normalize by batch size
        # norm /= b
        norm /= img_shape[0]
    if "c" in out_norm:
        # normalize by the number of channels
        # norm /= c
        norm /= img_shape[-3]
    if "i" in out_norm:
        # normalize by image/map size
        # norm /= h*w
        norm /= img_shape[-1] * img_shape[-2]

    return norm


def get_4dim_image_gradients(image: torch.Tensor):
    """
    Returns image gradients (dy, dx) for each color channel

    This uses the finite-difference approximation.
    Similar to get_image_gradients(), but additionally calculates the
    gradients in the two diagonal directions: 'dp' (the positive
    diagonal: bottom left to top right) and 'dn' (the negative
    diagonal: top left to bottom right).
    Only 1-step finite difference has been tested and is available.

    Args:
        image: Tensor with shape [b, c, h, w].

    Returns: tensors (dy, dx, dp, dn) holding the vertical, horizontal and
        diagonal image gradients (1-step finite difference). dx will
        always have zeros in the last column, dy will always have zeros
        in the last row, dp will always have zeros in the last row.

    """
    right = F.pad(image, (0, 1, 0, 0))[..., :, 1:]
    bottom = F.pad(image, (0, 0, 0, 1))[..., 1:, :]
    botright = F.pad(image, (0, 1, 0, 1))[..., 1:, 1:]

    dx, dy = right - image, bottom - image
    dn, dp = botright - image, right - bottom

    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0
    dp[:, :, -1, :] = 0

    return dx, dy, dp, dn


def get_image_gradients(image: torch.Tensor, step: int = 1):
    """
    Returns image gradients (dy, dx) for each color channel,

    This use the finite-difference approximation.
    Places the gradient [ie. I(x+1,y) - I(x,y)] on the base pixel (x, y).
    Both output tensors have the same shape as the input: [b, c, h, w].

    Args:
        image: Tensor with shape [b, c, h, w].
        step: the size of the step for the finite difference

    Returns: Pair of tensors (dy, dx) holding the vertical and horizontal
        image gradients (ie. 1-step finite difference). To match the
        original size image, for example with step=1, dy will always
        have zeros in the last row, and dx will always have zeros in
        the last column.

    """
    right = F.pad(image, (0, step, 0, 0))[..., :, step:]
    bottom = F.pad(image, (0, 0, 0, step))[..., step:, :]

    dx, dy = right - image, bottom - image

    dx[:, :, :, -step:] = 0
    dy[:, :, -step:, :] = 0

    return dx, dy


class TVLoss(nn.Module):
    """Calculate the L1 or L2 total variation regularization.

    Also can calculate experimental 4D directional total variation.
    Ref:
        Mahendran et al. https://arxiv.org/pdf/1412.0035.pdf
    """

    def __init__(
        self,
        tv_type: str = "tv",
        p=2,
        reduction: str = "mean",
        out_norm: str = "b",
        beta: int = 2,
    ) -> None:
        """
        Init

        Args:
            tv_type: regular 'tv' or 4D 'dtv'
            p: use the absolute values '1' or Euclidean distance '2' to
                calculate the tv. (alt names: 'l1' and 'l2')
            reduction: aggregate results per image either by their 'mean' or
                by the total 'sum'. Note: typically, 'sum' should be
                normalized with out_norm: 'bci', while 'mean' needs only 'b'.
            out_norm: normalizes the TV loss by either the batch size ('b'), the
                number of channels ('c'), the image size ('i') or combinations
                ('bi', 'bci', etc).
            beta: β factor to control the balance between sharp edges (1<β<2)
                and washed out results (penalizing edges) with β >= 2.
        """
        super(TVLoss, self).__init__()
        if isinstance(p, str):
            p = 1 if "1" in p else 2
        if p not in [1, 2]:
            raise ValueError(f"Expected p value to be 1 or 2, but got {p}")

        self.p = p
        self.tv_type = tv_type.lower()
        self.reduction = torch.sum if reduction == "sum" else torch.mean
        self.out_norm = out_norm
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method

        Args:
            x: data

        Returns: model outputs

        """
        norm = get_outnorm(x, self.out_norm)
        img_shape = x.shape
        if len(img_shape) == 3:
            # reduce all axes. (None is an alias for all axes.)
            reduce_axes = None
            _ = 1
        elif len(img_shape) == 4:
            # reduce for the last 3 axes.
            # results in a 1-D tensor with the tv for each image.
            reduce_axes = (-3, -2, -1)
            _ = x.size()[0]
        else:
            raise ValueError(
                "Expected input tensor to be of ndim " f"3 or 4, but got {len(img_shape)}"
            )

        if self.tv_type in ("dtv", "4d"):
            # 'dtv': dx, dy, dp, dn
            gradients = get_4dim_image_gradients(x)
        else:
            # 'tv': dx, dy
            gradients = get_image_gradients(x)

        # calculate the TV loss for each image in the batch
        loss = 0
        for grad_dir in gradients:
            if self.p == 1:
                loss += self.reduction(grad_dir.abs(), dim=reduce_axes)
            elif self.p == 2:
                loss += self.reduction(torch.pow(grad_dir, 2), dim=reduce_axes)

        # calculate the scalar loss-value for tv loss
        # Note: currently producing same result if 'b' norm or not,
        # but for some cases the individual image loss could be used
        loss = loss.sum() if "b" in self.out_norm else loss.mean()
        if self.beta != 2:
            loss = torch.pow(loss, self.beta / 2)

        return loss * norm
