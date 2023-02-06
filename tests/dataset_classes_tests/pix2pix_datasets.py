import torch
from datasets.classes.pix2pix_dataset import Pix2PixBaseDataset

class Pix2PixBaseDatasetTests:

    def __init__(self) -> None:
        self.dataset = Pix2PixBaseDataset(desired_img_size=64, orig_img_size=256)
        self.run_tests()
        return 

    def run_tests(self) -> None:
        self.test_1()
        return

    def test_1(self) -> None:
        print('\t\tTest 1')
        img_domA, img_domB = self.dataset[0]
        assert img_domA.size() == (3,64,64)
        assert img_domB.size() == (3,64,64)
        print('\t\tPassed.')
        return
