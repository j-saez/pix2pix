import torch
import torch.nn as nn
from utils.layers.disc_conv_layer import DiscConvBlock

"""
    Discriminator1x1: Discriminator model for images patches of size 1x1 pixels
    Inputs:
        >> in_chs: (int) Quantity of channels of the input image.
"""
class Discriminator1x1(nn.Module):
    def __init__(self, in_chs):
        super(Discriminator1x1, self).__init__()

        self.model = nn.Sequential(
            # As specified in the paper, Batch norm is not applied to the first c64 layer
            DiscConvBlock(in_chs*2,     out_chs=64,  padding_value=0, norm=False, patch_1x1=True),
            DiscConvBlock(in_chs=64,  out_chs=128, padding_value=0, norm=True, patch_1x1=True),
            # As specified in the paper, a convolution is applied to map a 1-dimensional ouptu followed by a sigmoid functions
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True, padding_mode='reflect'),
            nn.Sigmoid(),
        )
        return

    def forward(self, orig_img, transformed_img):
        _, _, orig_img_height, orig_img_width = orig_img.size()
        _, _, trans_img_height, trans_img_width = transformed_img.size()
        if (orig_img_height, orig_img_width) != (1, 1) and (trans_img_height, trans_img_width) != (1, 1):
            raise Exception(f'The input images need to be 1x1, and it has been fed an image of size IMG A: {orig_img_height}x{orig_img_width} || IMG B: {trans_img_height}x{trans_img_width}')

        x = torch.cat((orig_img, transformed_img),dim=1)
        return self.model(x)
