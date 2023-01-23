import torch
from models.discriminators.discriminator_patch16 import Discriminator16x16

B = 64
CHS = 3
H = 16
W = 16

class DiscPatch16Tests:
    def __init__(self) -> None:
        self.test_1()
        return

    def test_1(self) -> None:
        print('      Test 1.')
        x = torch.rand(B,CHS,H,W)
        model = Discriminator16x16(CHS)
        output = model(x,x)
        assert output.size() == (B, 1, 1, 1)
        print('      Passed.')
        return
