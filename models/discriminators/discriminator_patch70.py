import torch
import torch.nn as nn
from utils.layers.disc_conv_layer import DiscConvBlock

CHS_DIM = 1

"""
    Discriminator70x70: Discriminator model for images patches of size 70x70 pixels
"""
class Discriminator70x70(nn.Module):
    def __init__(self, in_chs):
        super(Discriminator70x70, self).__init__()
        self.model = nn.Sequential(
            # As specified in the paper, Batch norm is not applied to the first c64 layer
            DiscConvBlock(in_chs*2,   out_chs=64,  padding_value=1, norm=False),
            DiscConvBlock(in_chs=64,  out_chs=128, padding_value=1, norm=True),
            DiscConvBlock(in_chs=128, out_chs=256, padding_value=1, norm=True),
            DiscConvBlock(in_chs=256, out_chs=512, padding_value=1, norm=True),
            # As specified in the paper, a convolution is applied to map a 1-dimensional ouptu followed by a sigmoid functions
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='reflect'),
            nn.Sigmoid(),
        )
        return

    def forward(self, orig_img, transformed_img):
        _, _, orig_img_height, orig_img_width = orig_img.size()
        _, _, trans_img_height, trans_img_width = transformed_img.size()
        if (orig_img_height, orig_img_width) != (70, 70) and (trans_img_height, trans_img_width) != (70, 70):
            raise Exception(f'The input images need to be 70x70, and it has been fed an image of size IMG A: {orig_img_height}x{orig_img_width} || IMG B: {trans_img_height}x{trans_img_width}')

        x = torch.cat((orig_img, transformed_img), dim=CHS_DIM)
        return self.model(x)
