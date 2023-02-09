#############
## imports ## 
#############

from models.discriminators.discriminator_patch286 import Discriminator286x286
from models.discriminators.discriminator_patch70  import Discriminator70x70
from models.discriminators.discriminator_patch16  import Discriminator16x16
from models.discriminators.discriminator_patch1   import Discriminator1x1
from models.generators.generator_unet             import UNETGenerator
from models.generators.encoder_decoder_generator  import EncoderDecoderGenerator

###############
## functions ## 
###############

"""
    load_discriminator_model
    Description: Loads the discriminator model.
    Inputs:
        >> in_chs: (int) Total number of channels for the dataset to be fed.
        >> patch_size: (int) Patch size to select the correspondent discriminator.
    Outputs:
        >> model: (nn.Module) Discrimiantor model.
"""
def load_discriminator_model(in_chs, patch_size):
    model = []
    if patch_size==286:
        print('Loading Discriminator286x286')
        model = Discriminator286x286(in_chs)
    elif patch_size==70:
        print('Loading Discriminator70x70')
        model = Discriminator70x70(in_chs)
    elif patch_size==16:
        print('Loading Discriminator16x16')
        model = Discriminator16x16(in_chs)
    elif patch_size==1:
        print('Loading Discriminator1x1')
        model = Discriminator1x1(in_chs)
    else:
        raise Exception(f'There is no discriminator for {patch_size} patch size. Choose between: 286, 70, 16 and 1')
    return model

"""
    load_generator_model
    Description: Loads the generator model.
    Inputs:
        >> in_chs: (int) Total number of channels for the dataset to be fed.
        >> unet: (bool) Wheter to load the UNET generator or the encoder-decoder one
    Outputs:
        >> model: (nn.Module) Generator model.
"""
def load_generator_model(in_chs, unet):
    model = []
    if unet:
        print('Loading the UNET generator.')
        model = UNETGenerator(in_chs,)
    else:
        print('Loading the encoder-decoder generator.')
        model = EncoderDecoderGenerator(in_chs,)
    return model
