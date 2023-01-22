import torch.nn as nn
from utils.layers.disc_conv_layer import DiscConvBlock

"""
    Discriminator1x1: Discriminator model for images patches of size 1x1 pixels
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
        x = torch.cat((orig_img, transformed_img),dim=1)
        return self.model(x)

if __name__ == '__main__':
    
    import torch

    B = 64
    CHS = 3
    H = 1
    W = 1

    x = torch.rand(B,CHS,H,W)
    model = Discriminator1x1(CHS)
    output = model(x,x)

    print(f'output size = {output.size()}')
    assert output.size() == (B, 1, 1, 1)
