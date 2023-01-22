import torch.nn as nn

"""
    DiscConBlock: Convolutional block for the discriminator

    Following paper's instructions, all the convolutions have a 4x4 kernel size with stride 2
    and the discriminator downsample by a factor of 2.
"""

class DiscConvBlock(nn.Module):
    def __init__(self,in_chs, out_chs,padding_value,norm, patch_1x1=False):
        super(DiscConvBlock, self).__init__()
        kernel_size = 4 if not patch_1x1 else 1
        stride = 2 if not patch_1x1 else 1

        """
        Convolution + Batch norm + Relu or leaky relu
        """
        self.model = nn.Sequential()
        self.model.append( nn.Conv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride, padding=padding_value,bias=False, padding_mode='reflect') )
        if norm: self.model.append( nn.BatchNorm2d(out_chs) )
        self.model.append( nn.LeakyReLU(0.2) )

        return

    def forward(self, x):
        return self.model(x)
