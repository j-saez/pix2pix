import os
if os.getcwd()[-7:] != 'pix2pix':
    raise Exception('Run the file from the root directory.')

###########
# imports #
###########

import torchvision
from datasets.classes.pix2pix_dataset import Pix2PixBaseDataset

########################
## globals and consts ## 
########################

DATASETS_ORIG_SIZE = {'facades': 256, 'maps': 600, 'edges2shoes': 256, 'cityskapes': 256}
AVAILABLE_DATASETS = ['maps','facades','edges2shoes','cityscapes']
DATASETS_PATH = os.getcwd() + '/datasets/data/'

###############
## Functions ##
###############

"""
    Name: load_dataset
    Description: Return the train datasets of the selected dataset.
    Inputs:
        >> dataset_name: (str) Name of the dataset to be loaded.
        >> desired_img_size: (int) Desired size for squared image.
        >> val: (bool) Load validation or train data.
    Outputs:
        >> train_dataset: (torch.utils.data.Dataset) Train data.
"""
def load_dataset(dataset_name, desired_img_size, val, direction):
    dataset = None
    if not dataset_name in AVAILABLE_DATASETS:
        raise Exception(f'{dataset_name} not available. The avalable ones are: {AVAILABLE_DATASETS}')
    if dataset_name == 'maps':
        dataset = Pix2PixBaseDataset(desired_img_size, DATASETS_ORIG_SIZE['maps'],        val, DATASETS_PATH+'maps',       direction)
    elif dataset_name == 'edges2shoes':
        dataset = Pix2PixBaseDataset(desired_img_size, DATASETS_ORIG_SIZE['edges2shoes'], val, DATASETS_PATH+'edges2shoes',direction)
    elif dataset_name == 'facades':
        dataset = Pix2PixBaseDataset(desired_img_size, DATASETS_ORIG_SIZE['facades'],     val, DATASETS_PATH+'facades',    direction)
    elif dataset_name == 'cityscapes':
        dataset = Pix2PixBaseDataset(desired_img_size, DATASETS_ORIG_SIZE['cityscapes'],  val, DATASETS_PATH+'cityscapes', direction)
    else:
        raise Exception(f'{dataset_name} is not avalable. The avalable ones are: {AVAILABLE_DATASETS}')
    return dataset

