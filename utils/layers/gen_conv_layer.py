import torch.nn as nn

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
