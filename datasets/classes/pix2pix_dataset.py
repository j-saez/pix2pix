import os
from numpy import source
if os.getcwd()[-7:] != 'pix2pix':
    raise Exception('Run the file from the root directory.')

#############
## IMPORTS ##
#############

import torch
import albumentations
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, dataset
from PIL import Image

#######################
## globals and const ##
#######################

IMGS_CHS = 3
SOURCE_IMGS = 0
TARGET_IMGS = 1
NORMALIZE_TRANSFORMS = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize([0.5 for _ in range(IMGS_CHS)],[0.5 for _ in range(IMGS_CHS)],)
])

############# ## Classes ##
#############

"""

    Pix2PixBaseDataset
    Description: Class to load the datasets (maps, facades, edges2shoes, cityscapes) dowloaded from kaggle.
    Inputs:
        >> desired_img_size: (int) Desired size for the original image.
        >> orig_img_size: (int) Original size of squared image.
        >> train: (bool) Original size of squared image.
        >> dataset_path: Path to where the datasets to be loaded is saved.
    Outputs: None

"""

class Pix2PixBaseDataset(Dataset):

    def __init__(self, desired_img_size: int, orig_img_size: int, validation: bool, dataset_path: str, direction: str) -> None:
        self.direction = direction
        dataset_name = dataset_path.split('/')[-1]
        split = 'val' if validation else 'train'
        dataset_path = dataset_path + '/' + split + '/'

        # Previously loaded
        if os.path.isfile(f'./datasets/preloaded/{dataset_name}_{split}.pt'):
            data = torch.load(f'./datasets/preloaded/{dataset_name}_{split}.pt')
            self.source_dom_imgs = data[SOURCE_IMGS]
            self.target_dom_imgs = data[TARGET_IMGS]
        else:
            self.source_dom_imgs, self.target_dom_imgs = load_images(orig_img_size, dataset_path, desired_img_size)
            data = (self.source_dom_imgs, self.target_dom_imgs)
            torch.save(data, f'./datasets/preloaded/{dataset_name}_{split}.pt')
        return

    def __len__(self):
        return len(self.source_dom_imgs)

    def __getitem__(self, idx):
        if self.direction == "a_to_b":
            return  self.source_dom_imgs[idx], self.target_dom_imgs[idx]
        if self.direction == "b_to_a":
            return  self.source_dom_imgs[idx], self.target_dom_imgs[idx]
        raise Exception(f'Direction must be a_to_b or b_to_a, but received {self.direction}')

###############
## functions ##
###############

"""
    load_images
    Description: Load the images from the specified directory and resizes them to the desired size.
                 It expects images os size (CHS, HEIGHT, 2*WIDTH) as there have two be two images, the 
                 source one and the target one.
    Inputs:
        >> orig_size: (int) Orginal size of the squared images.
        >> dataset_path: (str)
        >> desired_size: (int) Desired size for squared images.
    Outputs:
        >> source_dom_imgs: torch tensor of size (total_images, imgs_chs, desired_size, desired_size) containing the source images.
        >> target_dom_imgs: torch tensor of size (total_images, imgs_chs, desired_size, desired_size) containing the target images.
"""

def load_images(orig_size, dataset_path, desired_size):
    print('\tLoading images...')
    # Init tensors
    total_images = len(os.listdir(dataset_path))
    source_dom_imgs = torch.zeros(total_images, IMGS_CHS, desired_size, desired_size)
    target_dom_imgs = torch.zeros(total_images, IMGS_CHS, desired_size, desired_size)

    # Load and resize images to desired size
    for idx, image_name in enumerate(os.listdir(dataset_path)):
        print(f'\t{idx}/{total_images}', end="\r")
        images_tensor   = NORMALIZE_TRANSFORMS(Image.open(dataset_path+image_name)).unsqueeze(dim=0)
        source_dom_imgs[idx, :, :, :] = F.interpolate(images_tensor[:, :, :, :orig_size], desired_size, mode='bilinear')
        target_dom_imgs[idx, :, :, :] = F.interpolate(images_tensor[:, :, :, orig_size:], desired_size, mode='bilinear')
    return source_dom_imgs, target_dom_imgs
