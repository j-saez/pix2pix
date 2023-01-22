import torch.nn as nn
from utils.layers.gen_conv_layer import GenDownsampling, GenUpsampling

"""
    EncoderDecoderGenerator: 
        Generator that does not use the UNET architecture as explained in the paper.
        Use at least an input of (Batch, Channels, 256, 256) as the input for the 
        forward method.
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

    """
        Use at least an input of size (Batch, Channels, 256, 256).
        In the original paper the size they use is the one above.
    """
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    import torch

    B = 64
    CHS = 3
    H = 256
    W = 256

    x = torch.rand(B,CHS,H,W)
    model = EncoderDecoderGenerator(CHS)
    output = model(x)

    print(f'output size = {output.size()}')
    assert output.size() == (B, CHS, H, W)
