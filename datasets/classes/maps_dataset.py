import os
if os.getcwd()[-7:] != 'pix2pix':
    raise Exception('Run the file from the root directory.')

import torch
form torch.utils.data import Dataset

DATASETS_PATH = os.getcwd() + '/datasets/'

class MapsDataset(Dataset):
    def __init__(self) -> None:
        return

    def __len__(self):
        return self.imgs_domA.size()[0]

    def __getitem__(self, idx):
        return self.imgs_domA[idx], self.imgs_domB[idx]
