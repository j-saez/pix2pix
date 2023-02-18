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
import torchvision
import torch.nn as nn
from utils.params                   import Params
from torch.utils.data.dataloader    import DataLoader
from torch.optim                    import Adam
from utils.datasets.loader          import load_dataset
from utils.training                 import load_tensorboard_writer, train_one_epoch, TrainingElements
from utils.model_selector           import load_discriminator_model, load_generator_model

###########################
## CONSTANTS AND GLOBALS ##
###########################

X_IMG = 0
Y_IMG = 1
DATASETS_CHS =       {'facades': 3,   'maps': 3,   'edges2shoes': 3,   'cityscapes': 3}
DATASETS_ORIG_SIZE = {'facades': 256, 'maps': 600, 'edges2shoes': 256, 'cityscapes': 256}

##########
## Main ##
##########

if __name__ == '__main__':

    print("PIX2PIX training loop")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} to train the model.".format(device))

    # Load params
    hyperparams, dataparams = Params().get_params()

    # Load the data
    val_dataset   = load_dataset(dataparams.dataset_name, desired_img_size=dataparams.img_size, val=True,  augmentation=dataparams.augmentation, direction=dataparams.direction)
    train_dataset = load_dataset(dataparams.dataset_name, desired_img_size=dataparams.img_size, val=False, augmentation=dataparams.augmentation, direction=dataparams.direction)
    train_dataloader = DataLoader(train_dataset,hyperparams.batch_size,shuffle=True)

    # Load the model
    disc = load_discriminator_model(DATASETS_CHS[dataparams.dataset_name], hyperparams.patch_size).to(device)
    gen = load_generator_model(DATASETS_CHS[dataparams.dataset_name], hyperparams.use_unet_gen).to(device)

    # Define optimizer and loss function
    disc_optimizer = Adam(disc.parameters(), hyperparams.lr, betas=(hyperparams.adam_beta1, hyperparams.adam_beta2))
    gen_optimizer = Adam(gen.parameters(), hyperparams.lr, betas=(hyperparams.adam_beta1, hyperparams.adam_beta2))
    bce_criterion = nn.BCELoss()
    l1_criterion = nn.L1Loss()

    # Create the tensorboard writer
    writer, weigths_folder = load_tensorboard_writer(hyperparams, dataparams.dataset_name)
    total_train_baches = int(len(train_dataset) / hyperparams.batch_size)

    # Create a TrainingElements object to reduce the number of arguments called in utils/training.py functions.
    training_elements = TrainingElements(
        train_dataloader,
        disc,
        gen,
        disc_optimizer,
        gen_optimizer,
        bce_criterion,
        l1_criterion,
        hyperparams.lr,
        0,
        hyperparams.total_epochs,
        total_train_baches,
        hyperparams.patch_size,
        hyperparams.l1_lambda,
        device,
        torchvision.transforms.RandomCrop(size=hyperparams.patch_size)
    )

    # Instead of fixed noise, we select fixed images to show in tensorboard
    rand_idx = torch.randint(low=0, high=len(val_dataset), size=(1,)).item()
    fixed_real_domA_img = train_dataset[rand_idx][X_IMG].unsqueeze(dim=0).to(device)
    fixed_real_domB_img = train_dataset[rand_idx][Y_IMG].unsqueeze(dim=0).to(device)
    step=0

    # Training loop
    print('\n\nStart of the training process.\n')
    for epoch in range(hyperparams.total_epochs):
        epoch_init_time = time.perf_counter()
        step = train_one_epoch(training_elements, writer, step, hyperparams.test_after_n_epochs)
        epoch_exec_time = epoch_init_time - time.perf_counter()

        if epoch % hyperparams.test_after_n_epochs == 0:
            # Test model
            with torch.no_grad():
                gen.eval()
                fake_domB_img = gen(fixed_real_domA_img)
                writer_images = torch.cat((fixed_real_domA_img.to('cpu'), fixed_real_domB_img.to('cpu'), fake_domB_img.to('cpu')), dim=0)
                writer.add_images(f'Real dom A | Real dom B | Fake dom B', writer_images.numpy(), epoch)
                step = step + 1
                torch.save(gen.state_dict(), weigths_folder+f'Generator_epoch_{epoch}.pt')
                torch.save(disc.state_dict(), weigths_folder+f'Discriminator_epoch_{epoch}.pt')
                gen.train()
    print('Training finished.')
