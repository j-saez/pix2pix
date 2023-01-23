import torch
from models.discriminators.discriminator_patch70 import Discriminator70x70

B = 64
CHS = 3
H = 70
W = 70

class DiscPatch70Tests:
    def __init__(self) -> None:
        self.test_1()
        return

    def test_1(self) -> None:
        print('      Test 1.')
        x = torch.rand(B,CHS,H,W)
        model = Discriminator70x70(CHS)
        output = model(x,x)
        assert output.size() == (B, 1, 1, 1)
        print('      Passed.')
        return
