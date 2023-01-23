import torch
from models.generators.encoder_decoder_generator import EncoderDecoderGenerator

B = 64
CHS = 3
H = 256
W = 256

class EncDecGenTests():
    def __init__(self) -> None:
        self.test_1()
        return

    def test_1(self) -> None:
        print('      Test 1.')
        x = torch.rand(B,CHS,H,W)
        model = EncoderDecoderGenerator(CHS)
        output = model(x)
        print(f'output size = {output.size()}')
        assert output.size() == (B, CHS, H, W)
        print('      Passed.')
        return
