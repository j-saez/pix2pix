import os
from numpy import source
if os.getcwd()[-7:] != 'pix2pix':
    raise Exception('Run the file from the root directory.')

#############
## IMPORTS ##
#############

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
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

#############
## Classes ##
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

    def __init__(self, desired_img_size: int, orig_img_size: int, validation: bool, dataset_path: str, augmentation: bool, direction: str) -> None:
        self.direction = direction
        dataset_name = dataset_path.split('/')[-1]
        split = 'val' if validation else 'train'
        dataset_path = dataset_path + '/' + split + '/'
        pre_loaded_filename = f'./datasets/preloaded/{dataset_name}_{split}_augmented.pt' if augmentation else f'./datasets/preloaded/{dataset_name}_{split}.pt'

        # Previously loaded
        if os.path.isfile(pre_loaded_filename):
            print(f'Loading dataset from {pre_loaded_filename}')
            data = torch.load(pre_loaded_filename)
            self.source_dom_imgs = data[SOURCE_IMGS]
            self.target_dom_imgs = data[TARGET_IMGS]
        else:
            self.source_dom_imgs, self.target_dom_imgs = load_images(orig_img_size, dataset_path, desired_img_size)
            if augmentation:
                self.source_dom_imgs, self.target_dom_imgs = augment_data(self.source_dom_imgs, self.target_dom_imgs)
            data = (self.source_dom_imgs, self.target_dom_imgs)
            print(f'Saving dataset file in {pre_loaded_filename} (This migth take a while).')
            torch.save(data, pre_loaded_filename)
            print('Saved.')

        print(f'\nTotal images SOURCE = {self.source_dom_imgs.size()[0]} || TARGET = {self.target_dom_imgs.size()[0]}\n')
        return

    def __len__(self):
        if self.source_dom_imgs.size()[0] != self.target_dom_imgs.size()[0]:
            raise Exception('Souce and target num of images mismatch.')
        return len(self.source_dom_imgs)

    def __getitem__(self, idx):
        if self.direction == "a_to_b":
            return  self.source_dom_imgs[idx], self.target_dom_imgs[idx]
        if self.direction == "b_to_a":
            return  self.target_dom_imgs[idx], self.source_dom_imgs[idx]
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
    print(f'total_image = {total_images}')
    source_dom_imgs = torch.zeros(total_images, IMGS_CHS, desired_size, desired_size)
    target_dom_imgs = torch.zeros(total_images, IMGS_CHS, desired_size, desired_size)

    # Load and resize images to desired size
    for idx, image_name in enumerate(os.listdir(dataset_path)):
        print(f'\t{idx}/{total_images}', end="\r")
        images_tensor   = NORMALIZE_TRANSFORMS(Image.open(dataset_path+image_name)).unsqueeze(dim=0)
        source_dom_imgs[idx, :, :, :] = F.interpolate(images_tensor[:, :, :, :orig_size], desired_size, mode='bilinear')
        target_dom_imgs[idx, :, :, :] = F.interpolate(images_tensor[:, :, :, orig_size:], desired_size, mode='bilinear')
    return source_dom_imgs, target_dom_imgs

"""
    augment_data
    Description: Augment the available data by rotating and flipping (vertically and horizontally).
                 Thus, if originally we had 10 images, the output will be 40 images (10 original + 10 rotated + 10 vflip + 10 hflip).
    Inputs:
        >> source_imgs: (torch tensor) Tensor with size (N, CH, H, W) containing the source images.
        >> target_imgs: (torch tensor) Tensor with size (N, CH, H, W) containing the target images.
    Outputs:
        >> source_imgs: (torch tensor) Tensor with size (4*N, CH, H, W) containing the source images.
        >> target_imgs: (torch tensor) Tensor with size (4*N, CH, H, W) containing the target images.
"""
def augment_data(source_imgs, target_imgs):
    print('\t Aumgenting images...')
    rot_angle = torch.randint(low=15, high=90, size=(1,)).item()
    # Source images transforms
    print('\t\t Source')
    aug_source_imgs = torch.cat((source_imgs, TF.hflip(source_imgs)))
    print('\t\t\t hflip')
    aug_source_imgs = torch.cat((aug_source_imgs, TF.vflip(source_imgs)))
    print('\t\t\t vflip')
    aug_source_imgs = torch.cat((aug_source_imgs, TF.rotate(source_imgs, angle=rot_angle)))
    print('\t\t\t rotation')
    print(f'aug_source_imgs size = {aug_source_imgs.size()}')

    # Target images transforms
    print('\t\t Target')
    aug_target_imgs = torch.cat((target_imgs, TF.hflip(target_imgs)))
    print('\t\t\t hflip')
    aug_target_imgs = torch.cat((aug_target_imgs, TF.vflip(target_imgs)))
    print('\t\t\t vflip')
    aug_target_imgs = torch.cat((aug_target_imgs, TF.rotate(target_imgs, angle=rot_angle)))
    print('\t\t\t rotation')
    return aug_source_imgs, aug_target_imgs
