import os
if os.getcwd()[-7:] != 'pix2pix':
    raise Exception('Run the file from the root directory.')

#############
## IMPORTS ##
#############

import torch
import torchvision
from torch.utils.data import Dataset
from utils.datasets.tools import load_images_names, transform_imgs
from PIL import Image

#######################
## globals and const ##
#######################

IMGS_CHS = 3
SOURCE_IMGS = 0
TARGET_IMGS = 1
DATA_AUG_TRANSFORMS = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize([0.5 for _ in range(IMGS_CHS)],[0.5 for _ in range(IMGS_CHS)],),
    torchvision.transforms.RandomRotation(degrees=(0,90)), 
    torchvision.transforms.RandomVerticalFlip(p=0.2),
    torchvision.transforms.RandomHorizontalFlip(p=0.2)
])

#############
## Classes ##
#############

"""

    Edges2ShoesDataset
    Description: Class to load the cityscapes datasets.
    Inputs:
        >> desired_img_size: (int) Desired size for the original image.
        >> orig_img_size: (int) Original size of squared image.
        >> train: (bool) Original size of squared image.
        >> dataset_path: Path to where the datasets to be loaded is saved.
    Outputs: None

"""

class Edges2ShoesDataset(Dataset):

    def __init__(self, validation: bool, dataset_path: str, direction: str) -> None:
        self.direction = direction
        split = 'val' if validation else 'train'
        dataset_path = dataset_path + '/' + split + '/'

        self.images_names = load_images_names(dataset_path)
        return

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        image = DATA_AUG_TRANSFORMS(Image.open(self.images_names[idx]))
        imgA = image[:,:,:256].view(3,256,256)
        imgB = image[:,:,256:].view(3,256,256)

        # Data aug
        if torch.rand(size=(1,)).item() < 0.2:
            print('Augment')
            imgA, imgB = transform_imgs(imgA, imgB)

        if self.direction == 'a_to_b':
            return imgA, imgB
        elif self.direction == 'b_to_a':
            return imgB, imgA
        raise Exception(f'Direction must be a_to_b or b_to_a and received {self.direction}.')
