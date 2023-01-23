import torch
from models.generators.generator_unet import UNETGenerator

B = 64
CHS = 3
H = 256
W = 256

class UNETGenTests():
    def __init__(self) -> None:
        self.test_1()
        return

    def test_1(self) -> None:
        print('      Test 1.')
        x = torch.rand(B,CHS,H,W)
        model = UNETGenerator(CHS)
        output = model(x)
        print(f'output size = {output.size()}')
        assert output.size() == (B, CHS, H, W)
        print('      Passed.')
        return
