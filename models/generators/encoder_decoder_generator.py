import torch.nn as nn
from utils.layers.gen_conv_layer import GenDownsampling, GenUpsampling

"""
    EncoderDecoderGenerator: 
        Generator that does use the encoder-decoder architecture explained in the paper.
    Inputs:
        >> in_chs: (int)
"""
class EncoderDecoderGenerator(nn.Module):
    def __init__(self, in_chs,) -> None:
        super(EncoderDecoderGenerator, self).__init__()
        self.model = nn.Sequential(
            # Encoder
            ## Following the paper, batch norm not applied to the first layer
            nn.Conv2d(in_chs, 64, kernel_size=4, stride=2, padding=1,padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            GenDownsampling(64,  128, padding_value=1,),
            GenDownsampling(128, 256, padding_value=1,),
            GenDownsampling(256, 512, padding_value=1,),
            GenDownsampling(512, 512, padding_value=1,),
            GenDownsampling(512, 512, padding_value=1,),
            GenDownsampling(512, 512, padding_value=1,),
            GenDownsampling(512, 512, padding_value=1,),

            # Decoder
            ## Following the paper, last layer followed by tanh
            GenUpsampling(512, 512, padding_value=1, dropout=True),
            GenUpsampling(512, 512, padding_value=1, dropout=True),
            GenUpsampling(512, 512, padding_value=1, dropout=True),
            GenUpsampling(512, 512, padding_value=1, dropout=False),
            GenUpsampling(512, 256, padding_value=1, dropout=False),
            GenUpsampling(256, 128, padding_value=1, dropout=False),
            GenUpsampling(128, 64, padding_value=1, dropout=False),
            nn.ConvTranspose2d(64, in_chs, kernel_size=4, stride=2, padding=1,),
            nn.Tanh(),
        )
        return

    def forward(self, x):
        _, _, height, width = x.size()
        if (height, width) != (256, 256):
            raise Exception(f'The input images need to be 256x256, and it has been fed an image of size {height}x{width}')
        return self.model(x)
