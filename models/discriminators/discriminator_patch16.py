import torch
import torch.nn as nn
from utils.layers.disc_conv_layer import DiscConvBlock

"""
    Discriminator16x16: Discriminator model for images patches of size 16x16 pixels
"""
class Discriminator16x16(nn.Module):
    def __init__(self, in_chs):
        super(Discriminator16x16, self).__init__()
        self.model = nn.Sequential(
            # As specified in the paper, Batch norm is not applied to the first c64 layer
            DiscConvBlock(in_chs*2,     out_chs=64,  padding_value=1, norm=False),
            DiscConvBlock(in_chs=64,  out_chs=128, padding_value=1, norm=True),
            # As specified in the paper, a convolution is applied to map a 1-dimensional ouptu followed by a sigmoid functions
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='reflect'),
            nn.Sigmoid(),
        )
        return

    def forward(self, orig_img, transformed_img):
        x = torch.cat((orig_img, transformed_img),dim=1)
        return self.model(x)
