import os
if os.getcwd()[-7:] != 'pix2pix':
    raise Exception('Run the file from the the root dir.\n')
import torch
import torchvision.transforms.functional as TVF

###############
## FUCNTIONS ##
###############

"""
    transform_imgs
    Description: Apply random rotatation, horizontal and vertical flip to source and target images.
    Inputs:
        >> imgA: (torch tensor) containing the source image.
        >> imgB: (torch tensor) containing the target image.
    Outputs:
        >> imgA: (torch tensor) containing the augmented source image.
        >> imgB: (torch tensor) containing the augmented target image.
"""
def transform_imgs(imgA, imgB):
    rot_angle = float(torch.randint(low=0, high=45,size=(1,)).item())

    imgA = TVF.rotate(imgA, angle=rot_angle)
    imgA = TVF.hflip(imgA)
    imgA = TVF.vflip(imgA)

    imgB = TVF.rotate(imgB, angle=rot_angle)
    imgB = TVF.hflip(imgB)
    imgB = TVF.vflip(imgB)

    return imgA, imgB

"""
    load_images_names
    Description: Load the names of the iamges from the train dir in the document.
    Inputs:
        >> dataset_path: (str)
    Outputs:
        >> images_names: (list) containing the names of the images.
"""

def load_images_names(dataset_path):
    images_names = []
    for image_name in os.listdir(dataset_path):
        images_names.append(dataset_path+'/'+image_name)
    return images_names
