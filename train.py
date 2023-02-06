import os
if os.getcwd()[-7:] != 'pix2pix':
    message = 'Run the file from the the root dir:\n'
    message += 'cd pix2pix\n'
    message += 'python train.py'
    raise Exception(message)

############
## IMPORT ##
############

import time
import torch
import torch.nn as nn
from utils.params                   import Params
from torch.utils.data.dataloader    import DataLoader
from torch.optim                    import Adam
from utils.datasets                 import load_dataset
from utils.training                 import load_tensorboard_writer, train_one_epoch
from utils.model_selector           import load_discriminator_model, load_generator_model

###########################
## CONSTANTS AND GLOBALS ##
###########################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} to train the model.".format(DEVICE))

###########################
## Classes and functions ##
###########################

DATASETS_CHS =       {'facades': 3,   'maps': 3,   'edges2shoes': 3,   'cityskapes': 3}
DATASETS_ORIG_SIZE = {'facades': 256, 'maps': 600, 'edges2shoes': 256, 'cityskapes': 256}

##########
## Main ##
##########

if __name__ == '__main__':

    print("PIX2PIX training loop")

    # Load params
    print("Loading params")
    hyperparms, dataparams = Params().get_params()
    print("\tParams loaded.")

    # Load the data
    print("Loading dataset")
    img_size = (dataparams.img_size,dataparams.img_size)
    train_dataset = load_dataset(dataparams.dataset_name, desired_img_size=256)
    train_dataloader = DataLoader(train_dataset,hyperparms.batch_size,shuffle=True)
    print("\tDataset loaded.")

    # Load the model
    print("Loading the models")
    disc = load_discriminator_model(DATASETS_CHS[dataparams.dataset_name], hyperparms.patch_size).to(DEVICE)
    gen = load_generator_model(DATASETS_CHS[dataparams.dataset_name], hyperparms.use_unet_gen).to(DEVICE)
    print("\tModels loaded.")

    # Define optimizer and loss function
    print("Selecting optimizer")
    disc_optimizer = Adam(disc.parameters(), hyperparms.lr, betas=(hyperparms.adam_beta1, hyperparms.adam_beta2))
    gen_optimizer = Adam(gen.parameters(), hyperparms.lr, betas=(hyperparms.adam_beta1, hyperparms.adam_beta2))
    bce_criterion = nn.BCELoss()
    l1_criterion = nn.L1Loss()
    print("\tDone.")

    writer, weigths_folder = load_tensorboard_writer(hyperparms, dataparams.dataset_name)
    total_train_baches = int(len(train_dataset) / hyperparms.batch_size)

    # Instead of fixed noise, we select a fixed img from dom A
    fixed_real_img = train_dataset[0][0].unsqueeze(dim=0).to(DEVICE)
    print(f'fixed_real_img shape = {fixed_real_img.size()}')
    step = 0

    # Training loop
    print('\n\nStart of the training process.\n')
    for epoch in range(hyperparms.total_epochs):

        epoch_init_time = time.perf_counter()

        step = train_one_epoch(train_dataloader,disc,gen,disc_optimizer,gen_optimizer,bce_criterion,l1_criterion,hyperparms,epoch,total_train_baches,DEVICE,writer,step)
        epoch_exec_time = epoch_init_time - time.perf_counter()

        if epoch % hyperparms.test_after_n_epochs == 0:
            # Test model
            with torch.no_grad():
                gen.eval()
                test_generated_imgs = gen(fixed_real_img).to('cpu')
                writer.add_images(f'Generated_images', test_generated_imgs.numpy(), epoch+1)
                step = step + 1
                torch.save(gen.state_dict(), weigths_folder+f'Generator_epoch_{epoch}.pt')
                torch.save(disc.state_dict(), weigths_folder+f'Discriminator_epoch_{epoch}.pt')
                gen.train()
    print('Training finished.')
