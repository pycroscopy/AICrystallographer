# -*- coding: utf-8 -*-
"""
Custom blocks for convolutional neural networks

@author: Maxim Ziatdinov

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Builds a residual block
    """
    def __init__(self, nb_filters_in=20, nb_filters_out=40, lrelu_a=0):
        """
        Args:
            nb_filters_in (int): number of channels in the block input
            nb_filters_out (int): number of channels in the block output
            lrelu_a=0 (float): negative slope value for leaky ReLU
        """
        super(ResBlock, self).__init__()
        self.lrelu_a = lrelu_a
        self.c0 = nn.Conv2d(nb_filters_in,
                            nb_filters_out,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.c1 = nn.Conv2d(nb_filters_out,
                           nb_filters_out,
                           kernel_size=3,
                           stride=1,
                           padding=1)
        self.c2 = nn.Conv2d(nb_filters_out,
                           nb_filters_out,
                           kernel_size=3,
                           stride=1,
                           padding=1)
        self.bn1 = nn.BatchNorm2d(nb_filters_out)
        self.bn2 = nn.BatchNorm2d(nb_filters_out)

    def forward(self, x):
        """Defines forward path"""
        x = self.c0(x)
        residual = x
        out = self.c1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out, negative_slope=self.lrelu_a)
        out = self.c2(out)
        out = self.bn2(out)
        out += residual
        out = F.leaky_relu(out, negative_slope=self.lrelu_a)
        return out


class ConvEncoder(nn.Module):
    """
    Builds a convolutional encoder
    """
    def __init__(self, nb_filters=16, lrelu_a=0):
        """
        Args:
            nb_filters (int): number of filters in the first layer
            lrelu_a (float): negative slope for leaky ReLU activation
        """
        super(ConvEncoder, self).__init__()
        self.lrelu_a = lrelu_a
        self.c1 = nn.Conv2d(1,
                            nb_filters,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.c2 = nn.Conv2d(nb_filters,
                            nb_filters*2,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.c3 = nn.Conv2d(nb_filters*2,
                            nb_filters*2,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.bn1 = nn.BatchNorm2d(nb_filters)
        self.bn2 = nn.BatchNorm2d(nb_filters*2)
        self.bn3 = nn.BatchNorm2d(nb_filters*2)

    def forward(self, x):
        """Forward path"""
        x = self.c1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=self.lrelu_a)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.c2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=self.lrelu_a)
        x = self.c3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=self.lrelu_a)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x


class ConvDecoder(nn.Module):
    """ Builds convolutional decoder"""
    def __init__(self, nb_filters=32, lrelu_a=0):
        """
        Args:
            nb_filters (int): number of filters in the first layer
            lrelu_a (float): negative slope for leaky ReLU activation
        """
        super(ConvDecoder, self).__init__()
        self.lrelu_a = lrelu_a
        self.d1 = nn.Conv2d(nb_filters,
                            nb_filters,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.d2 = nn.Conv2d(nb_filters,
                            nb_filters,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.d3 = nn.Conv2d(nb_filters,
                            nb_filters//2,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.bn1 = nn.BatchNorm2d(nb_filters)
        self.bn2 = nn.BatchNorm2d(nb_filters)
        self.bn3 = nn.BatchNorm2d(nb_filters//2)

    def forward(self, x):
        """Forward path"""
        x = self.d1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=self.lrelu_a)
        x = self.d2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=self.lrelu_a)
        x = F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.d3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=self.lrelu_a)
        x = F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=False)
        return x
