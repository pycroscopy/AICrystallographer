# -*- codisg: utf-8 -*-
"""
Constructor (and loader) of fully convolutional neural network

@author: Maxim Ziatdinov

"""

from nnblocks import *


def load_torchmodel(weights_path, model):
    """
    Loads saved weights into a model
    """
    if torch.cuda.device_count() > 0:
        checkpoint = torch.load(weights_path)
    else:
        checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model


class Sniffer(nn.Module):
    """
    Convolutional neural network with an 'encoder-decoder' architecture
    """
    def __init__(self, nb_filters_en=16, lrelu_a=0, with_logits=False):
        """
        Args:
            nb_filters_en (int): number of filters in the 1st layer of encoder
            lrelu_a (float): negative slope in leaky ReLU
            with_logits(bool): determines if output of a final
            convolutional layer is passed/not passed through a sigmoid
            activation
        """
        super(Sniffer, self).__init__()
        self.with_logits = with_logits
        self.encoder = ConvEncoder(nb_filters=nb_filters_en, lrelu_a=lrelu_a)
        self.decoder = ConvDecoder(nb_filters=nb_filters_en*2, lrelu_a=lrelu_a)
        self.last_layer = nn.Conv2d(nb_filters_en, 1, kernel_size=3,
                                    stride=1, padding=1)

    def forward(self, x):
        """Defines a forward path"""
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.last_layer(x)
        if self.with_logits:
            return x
        return torch.sigmoid(x)


class ResSniffer(nn.Module):
    """
    Convolutional neural network with residual blocks
    """
    def __init__(self, input_channels=32, res_depth=4, lrelu_a=0, with_logits=False):
        """
        Args:
            nb_filters (int): number of filters in the 1st layer
            lrelu_a (float): negative slope in leaky ReLU
            with_logits (bool): determines if output of a final
            convolutional layer is passed/not passed through a sigmoid
            activation
        """
        super(ResSniffer, self).__init__()
        output_channels = input_channels*2
        self.with_logits = with_logits
        # Define first layer
        self.conv_in = nn.Sequential(
            nn.Conv2d(1, input_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=lrelu_a, inplace=True),
            nn.BatchNorm2d(input_channels))
        # Define pre-last layer
        self.conv_out = nn.Sequential(
            nn.Conv2d(output_channels, input_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=lrelu_a, inplace=True),
            nn.BatchNorm2d(input_channels))
        # Define final layer (1-by-1 convolution)
        self.conv_px = nn.Conv2d(input_channels, 1, kernel_size=1, stride=1, padding=0)
        # Define residual block
        res_module = []
        for i in range(res_depth):
            input_channels = output_channels if i > 0 else input_channels
            res_module.append(
                ResBlock(input_channels, output_channels, lrelu_a=lrelu_a))
        self.res_module = nn.Sequential(*res_module)

    def forward(self, x):
        """Defines a forward path """
        x = self.conv_in(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.res_module(x)
        x = F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv_out(x)
        x = self.conv_px(x)
        if self.with_logits:
            return x
        return torch.sigmoid(x)
