from models.discriminator.discriminator_patch286 import Discriminator286x286
from models.discriminator.discriminator_patch70  import Discriminator70x70
from models.discriminator.discriminator_patch16  import Discriminator16x16
from models.discriminator.discriminator_patch1   import Discriminator1x1
from models.generator.unet_generator             import UNETGenerator
from models.generator.no_skip_generator          import EncoderDecoderGenerator

def load_discriminator_model(in_chs, patch_size):
    model = []
    if patch_size==286:
        model = Discriminator286x286(in_chs)
    elif patch_size==70:
        model = Discriminator70x70(in_chs)
    elif patch_size==16:
        model = Discriminator16x16(in_chs)
    elif patch_size==1:
        model = Discriminator1x1(in_chs)
    else:
        raise Exception(f'There is no discriminator for {} patch size. Choose between: 286, 70, 16 and 1')
    return model

def load_generator_model(in_chs, unet):
    model = []
    if unet:
        model = UNETGenerator(in_chs,)
    else:
        model = EncoderDecoderGenerator(in_chs,)
    return model
