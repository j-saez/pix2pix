import os
if os.getcwd()[-7:] != 'pix2pix':
    message = 'Run the file from the the root dir:\n'
    message += 'cd pix2pix\n'
    message += 'python inference.py'
    raise Exception(message)

############
## IMPORT ##
############

import cv2
import torch
import argparse
from utils.params                   import Params
from utils.datasets                 import load_dataset
from utils.model_selector           import load_discriminator_model, load_generator_model

#############
## GLOBALS ##
#############

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############
## FUNCTIONS ##
###############

"""
    load_args
    description: Load the arguments for the script.
    Inputs: None
    Outputs: 
        >> args: (ArgumentParser args) Arguments passed for the script.
"""
def load_args():
    parser = argparse.ArgumentParser(description='Arguments for pix2pix inference.')
    parser.add_argument( '--generator',    type=str, required=True, help='Generator model. Choose between unet or encoder-decoder' )
    parser.add_argument( '--model_weigths', type=str, required=True, help='Path to the model weigts to be loaded.' )
    parser.add_argument( '--image', type=str, required=True, help='Path to the image.' )

    args = parser.parse_args()
    check_arguments(args)
    return args

"""
    check_arguments
    description: Checks if the arguments passed can be found or are correct. If not an Exception raised.
    Inputs:
        >> args: (ArgumentParser args) Arguments passed for the script.
    Outputs: None
"""
def check_arguments(args) -> None:
    generator_name = args.generator_name
    if generator_name != 'unet' and generator_name != 'encoder-decoder':
        raise Exception(f'The model {generator_name} is not available. Choose between unet or encoder-decoder.')

    generator_weights = args.model_weigths
    if not os.path.isfile(generator_weights):
        raise Exception(f'Weights file {generator_weights} not found. Make sure you have added the hole path.')

    img_name = args.image
    if not os.path.isfile(img_name):
        raise Exception(f'Image file {img_name} not found. Make sure you have added the hole path.')
    return

##########
## MAIN ##
##########

if __name__ == '__main__':

    # Load args
    args = load_args()
    generator_name = args.generator_name
    generator_weights = args.model_weigths
    img_name = args.image

    # Load image
    numpy_img = cv2.imread(img_name)
    torch_img = torch.from_numpy(numpy_img).unsqueeze(dim=0)
    print(f'torch_img size = {torch_img.size()}')
    _, in_chs, _, _ = torch_img.size()
    img_name = args.image

    # Load generator
    load_unet = True if generator_name == 'unet' else 'encoder-decoder'
    generator = load_generator_model(in_chs, load_unet).to(DEVICE).eval()
