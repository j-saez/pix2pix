import os
if os.getcwd()[-7:] != 'pix2pix':
    raise Exception('Run the file from the root directory.')

#############
## IMPORTS ##
#############

import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset

#######################
## globals and const ##
#######################

DATASETS_PATH = os.getcwd() + '/datasets/data/'
IMGS_CHS = 3

#############
## Classes ##
#############

"""

    Pix2PixBaseDataset
    Description: Class to load the datasets (maps, facades, edges2shoes, cityscapes) dowloaded from kaggle.
    Inputs:
        >> desired_img_size: (int) Desired size for the original image.
        >> orig_img_size: (int) Original size of squared image.
        >> dataset_path: Path to where the datasets to be loaded is saved.
    Outputs: None

"""
class Pix2PixBaseDataset(Dataset):
    def __init__(self, desired_img_size, orig_img_size, dataset_path=DATASETS_PATH) -> None:
        self.desired_img_size = desired_img_size
        self.orig_img_size = orig_img_size 
        self.imgs_folder = torchvision.datasets.ImageFolder(dataset_path)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize([0.5 for _ in range(IMGS_CHS)],[0.5 for _ in range(IMGS_CHS)],)
        ])
        return

    def __len__(self):
        return len(self.imgs_folder)

    def __getitem__(self, idx):
        orig_img = self.transforms(self.imgs_folder[idx][0]).unsqueeze(dim=0)
        img_domA = orig_img[:,:, :, :self.orig_img_size]
        img_domB = orig_img[:,:, :, self.orig_img_size:]

        # Resize images
        img_domA = F.interpolate(img_domA, size=self.desired_img_size, mode='bilinear').view(IMGS_CHS,self.desired_img_size,self.desired_img_size)
        img_domB = F.interpolate(img_domB, size=self.desired_img_size, mode='bilinear').view(IMGS_CHS,self.desired_img_size,self.desired_img_size)
        return  img_domA, img_domB
