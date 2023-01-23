import torch
from models.discriminators.discriminator_patch1 import Discriminator1x1

B = 64
CHS = 3
H = 1
W = 1

class DiscPatch1Tests:
    def __init__(self) -> None:
        self.test_1()
        return

    def test_1(self) -> None:
        print('      Test 1.')
        x = torch.rand(B,CHS,H,W)
        model = Discriminator1x1(CHS)
        output = model(x,x)
        assert output.size() == (B, 1, 1, 1)
        print('      Passed.')
        return
