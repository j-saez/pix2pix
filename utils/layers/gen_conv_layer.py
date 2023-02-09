import torch.nn as nn

"""
    GenDownsampling class
    Class description: Return a layer for the downsampling part of the generator.
                       This layer is composed by a 2D convolution, batch normalization
                       and Leaky ReLU with 0.2 slope as described in the paper.
    Inputs:
        >> in_chs: (int) Channels that the layer is going to receive as input.
        >> out_chs: (int) Channels that the layer is going to ouput.
        >> padding_value: (int)
    Outputs:
        >> layer: nn.Module for a layer in the downsampling part of the generator.
"""
class GenDownsampling(nn.Module):
    def __init__(self,in_chs, out_chs,padding_value,):
        super(GenDownsampling, self).__init__()
        """
        Following paper for the encoder:
            Convolution (4x4 kernel_size and stride 2) + Batch norm + leaky relu
        """
        self.model = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=4, stride=2, padding=padding_value,bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_chs),
            nn.LeakyReLU(0.2)
        )
        return

    def forward(self, x):
        return self.model(x)

"""
    GenUpsampling class
    Class description: Return a layer for the upsampling part of the generator.
                       This layer is composed by a 2D transpose convolution, batch normalization
                       and ReLU and (if specified) batch normalization slope as described in the paper.
    Inputs:
        >> in_chs: (int) Channels that the layer is going to receive as input.
        >> out_chs: (int) Channels that the layer is going to ouput.
        >> padding_value: (int) Value for padding in the 2D conv.
        >> droput: (bool) Whether to apply dropout or not in the layer.
    Outputs:
        >> layer: nn.Module for a layer in the downsampling part of the generator.
"""
class GenUpsampling(nn.Module):
    def __init__(self,in_chs, out_chs,padding_value,dropout):
        super(GenUpsampling, self).__init__()
        """
        Following paper, for the decoder:
            TransposeConv2d (4x4 kernel_size and stride 2) + Batch norm + Relu + dropout if specified
        """
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_chs, out_chs, kernel_size=4, stride=2, padding=padding_value,bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU()
        )
        if dropout:
            self.model.append(nn.Dropout(0.5))
        return

    def forward(self, x):
        return self.model(x)
