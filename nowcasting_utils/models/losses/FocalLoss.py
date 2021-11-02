""" Focal Loss - https://arxiv.org/abs/1708.02002 """
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable


class FocalLoss(nn.Module):
    """Focal Loss"""

    def __init__(
        self,
        gamma: Union[int, float, List] = 0,
        alpha: Optional[Union[int, float, List]] = None,
        size_average: bool = True,
    ):
        """
        Focal loss is described in https://arxiv.org/abs/1708.02002

        Copied from: https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py

        Courtesy of carwin, MIT License

        Args:
            alpha: (tensor, float, or list of floats) The scalar factor for this criterion
            gamma: (float,double) gamma > 0 reduces the relative loss for well-classified
                examples (p>0.5) putting more focus on hard misclassified example
            size_average: (bool, optional) By default, the losses are averaged over
                each loss element in the batch.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        """
        Forward model

        Args:
            x: prediction
            target: truth

        Returns: loss value

        """
        if x.dim() > 2:
            x = x.view(x.size(0), x.size(1), -1)  # N,C,H,W => N,C,H*W
            x = x.transpose(1, 2)  # N,C,H*W => N,H*W,C
            x = x.contiguous().view(-1, x.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(x)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != x.data.type():
                self.alpha = self.alpha.data.type_as(x)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
