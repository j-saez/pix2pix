import torch
import torch.nn as nn
from utils.layers.gen_conv_layer import GenDownsampling, GenUpsampling

"""
    UNETGenerator
    Description: Class for the UNET discriminator used in the Pix2Pix paper.
    Inputs:
        >> in_chs: (int) Quantity of channels of the input images.
"""
class UNETGenerator(nn.Module):

    def __init__(self, in_chs: int,) -> None:
        super().__init__()

        # Encoder
        ## Following the paper, batch norm not applied to the first layer
        self.enc_input_layer = nn.Sequential( # 128 x 128 
            nn.Conv2d(in_chs, 64, kernel_size=4, stride=2, padding=1,padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        self.enc_l1 = GenDownsampling(64,  128, padding_value=1,) # 64 x 64
        self.enc_l2 = GenDownsampling(128, 256, padding_value=1,) # 32 x 32
        self.enc_l3 = GenDownsampling(256, 512, padding_value=1,) # 16 x 16
        self.enc_l4 = GenDownsampling(512, 512, padding_value=1,) # 8 x 8
        self.enc_l5 = GenDownsampling(512, 512, padding_value=1,) # 4 x 4
        self.enc_l6 = GenDownsampling(512, 512, padding_value=1,) # 2 x 2
        self.enc_l7 = GenDownsampling(512, 512, padding_value=1,) # 1 x 1

        # Decoder
        self.dec_l1 = GenUpsampling(512,   512, padding_value=1, dropout=True) # 2 x 2
        self.dec_l2 = GenUpsampling(512*2, 512, padding_value=1, dropout=True) # 4 x 4
        self.dec_l3 = GenUpsampling(512*2, 512, padding_value=1, dropout=True) # 8 x 8
        self.dec_l4 = GenUpsampling(512*2, 512, padding_value=1, dropout=True) # 16 x 16
        self.dec_l5 = GenUpsampling(512*2, 256, padding_value=1, dropout=True) # 32 x 32
        self.dec_l6 = GenUpsampling(256*2, 128, padding_value=1, dropout=True) # 64 x 64
        self.dec_l7 = GenUpsampling(128*2, 64,  padding_value=1, dropout=True) # 128 x 128
        self.dec_ouptput_layer = nn.Sequential( # 256 x 256
            nn.ConvTranspose2d(64*2, in_chs, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        return

    def forward(self, x):
        _, _, height, width = x.size()
        if (height, width) != (256, 256):
            raise Exception(f'The input images need to be 256x256, and it has been fed an image of size {height}x{width}')

        # Encoder
        enc_input_layer_out = self.enc_input_layer(x)
        enc_l1_out = self.enc_l1(enc_input_layer_out)
        enc_l2_out = self.enc_l2(enc_l1_out)
        enc_l3_out = self.enc_l3(enc_l2_out)
        enc_l4_out = self.enc_l4(enc_l3_out)
        enc_l5_out = self.enc_l5(enc_l4_out)
        enc_l6_out = self.enc_l6(enc_l5_out)
        enc_l7_out = self.enc_l7(enc_l6_out)

        # Decoder
        dec_l1_out = self.dec_l1(enc_l7_out)
        print(f'size = {dec_l1_out.size()}')
        dec_l2_out = self.dec_l2(torch.cat((dec_l1_out, enc_l6_out),dim=1))
        print(f'size = {dec_l2_out.size()}')
        dec_l3_out = self.dec_l3(torch.cat((dec_l2_out, enc_l5_out),dim=1))
        print(f'size = {dec_l3_out.size()}')
        dec_l4_out = self.dec_l4(torch.cat((dec_l3_out, enc_l4_out),dim=1))
        print(f'size = {dec_l4_out.size()}')
        dec_l5_out = self.dec_l5(torch.cat((dec_l4_out, enc_l3_out),dim=1))
        print(f'size = {dec_l5_out.size()}')
        dec_l6_out = self.dec_l6(torch.cat((dec_l5_out, enc_l2_out),dim=1))
        print(f'size = {dec_l6_out.size()}')
        dec_l7_out = self.dec_l7(torch.cat((dec_l6_out, enc_l1_out),dim=1))
        print(f'size = {dec_l7_out.size()}')
        output = self.dec_ouptput_layer(torch.cat((dec_l7_out, enc_input_layer_out),dim=1))
        print(f'size = {output.size()}')

        return output
