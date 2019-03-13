# -*- coding: utf-8 -*-
"""
Custom focal loss function for pytorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class focal(nn.Module):
    """
    Loss function for classification tasks  with
    large data imbalance. Focal loss (FL) is define as:
    FL(p_t) = -alpha*((1-p_t)^gamma))*log(p_t),
    where p_t is a cross-entropy loss for binary classification.
    For more details, see https://arxiv.org/abs/1708.02002.
    """
    def __init__(self, alpha=0.5, gamma=2, with_logits=True):
        """
        Args:
            alpha (float): "balance" coefficient,
            gamma (float): "focusing" parameter (>=0),
            with_logits (bool): indicates if the sigmoid operation was applied
            at the end of a neural network's forward path.
        """
        super(focal, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = with_logits

    def forward(self, images, labels):
        """Calculate focal loss"""
        if self.logits:
            CE_loss = F.binary_cross_entropy_with_logits(images, labels)
        else:
            CE_loss = F.binary_cross_entropy(images, labels)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss
        return F_loss
