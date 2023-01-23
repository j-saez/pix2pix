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
from utils.datasets                 import load_dataset, get_dataset_transforms
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

DATASETS_CHS = {'mnist': 1}

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
    transforms = get_dataset_transforms(img_size, DATASETS_CHS[dataparams.dataset_name])
    train_dataset = load_dataset(dataparams.dataset_name, transforms)
    train_dataloader = DataLoader(train_dataset,hyperparms.batch_size,shuffle=True)
    print("\tDataset loaded.")

    # Load the model
    print("Loading the models")
    (img_h, img_w) = img_size
    norm = [True,True,True,True]
    disc = load_discriminator_model(DATASETS_CHS[dataparams.dataset_name], hyperparms.patch_size)
    gen = load_generator_model(DATASETS_CHS[dataparams.dataset_name], hyperparms.use_unet_gen)
    print("\tModels loaded.")

    # Define optimizer and loss function
    print("Selecting optimizer")
    disc_optimizer = Adam(disc.parameters(), hyperparms.lr, betas=(hyperparms.adam_beta1, hyperparms.adam_beta2))
    gen_optimizer = Adam(gen.parameters(), hyperparms.lr, betas=(hyperparms.adam_beta1, hyperparms.adam_beta2))
    criterion = nn.BCELoss()
    print("\tDone.")

    writer, weigths_folder = load_tensorboard_writer(hyperparms, dataparams.dataset_name, norm)
    total_train_baches = int(len(train_dataset) / hyperparms.batch_size)
    # Instead of fixed noise, select a fixed real_img
    #fixed_noise = torch.rand(hyperparms.batch_size, hyperparms.z_dim,1,1).to(DEVICE)
    step = 0

    # Training loop
    print('\n\nStart of the training process.\n')
    for epoch in range(hyperparms.total_epochs):

        epoch_init_time = time.perf_counter()

        step = train_one_epoch(train_dataloader,disc,gen,disc_optimizer,gen_optimizer,criterion,hyperparms,epoch,total_train_baches,DEVICE,writer,step)
        epoch_exec_time = epoch_init_time - time.perf_counter()

        if epoch % hyperparms.test_after_n_epochs == 0:
            # Test model
            with torch.no_grad():
                test_generated_imgs = gen(fixed_noise)[:16,:,:,:].to('cpu')
                writer.add_images(f'Generated_images', test_generated_imgs.numpy(), epoch+1)
                step = step + 1
                torch.save(gen.state_dict(), weigths_folder+f'Generator_epoch_{epoch}.pt')
                torch.save(disc.state_dict(), weigths_folder+f'Discriminator_epoch_{epoch}.pt')
    print('Training finished.')
