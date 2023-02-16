import os
from numpy import source
if os.getcwd()[-7:] != 'pix2pix':
    raise Exception('Run the file from the root directory.')

#############
## IMPORTS ##
#############

import torch
import torchvision
from torch.utils.data      import Dataset
from utils.datasets.tools  import transform_imgs, load_images_names
from PIL import Image

#######################
## globals and const ##
#######################

IMGS_CHS = 3
SOURCE_IMGS = 0
TARGET_IMGS = 1
NORM_TRANSFORMS = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize([0.5 for _ in range(IMGS_CHS)],[0.5 for _ in range(IMGS_CHS)],),
])

#############
## Classes ##
#############

"""

TODO

    CityScapesDataset
    Description: Class to load the cityscapes dataset.
    Inputs:
        >> validation: (bool) whether to load the training or validation set.
        >> dataset_path: (str) Path where the dataset is located.
        >> direction: (str) Indicating if the model is going to be trained to translate from dom A to dom B or viceversa.
    Outputs: None

"""

class CityScapesDataset(Dataset):

    def __init__(self, validation: bool, dataset_path: str, direction: str) -> None:
        self.direction = direction
        split = 'val' if validation else 'train'
        dataset_path = dataset_path + '/' + split + '/'

        self.images_names = load_images_names(dataset_path)
        return

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        image = NORM_TRANSFORMS(Image.open(self.images_names[idx]))
        imgA = image[:,:,:256].view(3,256,256)
        imgB = image[:,:,256:].view(3,256,256)

        # Data aug
        if torch.rand(size=(1,)).item() < 0.2:
            imgA, imgB = transform_imgs(imgA, imgB)

        if self.direction == 'a_to_b':
            return imgA, imgB
        elif self.direction == 'b_to_a':
            return imgB, imgA
        raise Exception(f'Direction must be a_to_b or b_to_a and received {self.direction}.')
