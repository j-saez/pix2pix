import torch
from models.discriminators.discriminator_patch286 import Discriminator286x286

B = 64
CHS = 3
H = 256
W = 256

class DiscPatch256Tests:
    def __init__(self) -> None:
        self.test_1()
        return

    def test_1(self) -> None:
        print('      Test 1.')
        x = torch.rand(B,CHS,H,W)
        model = Discriminator286x286(CHS)
        output = model(x,x)
        assert output.size() == (B, 1, 1, 1)
        print('      Passed.')
        return
