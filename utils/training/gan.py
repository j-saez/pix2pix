import os
if os.getcwd()[-7:] != 'pix2pix':
    message = 'Run the file from the the root dir:\n'
    message += 'cd pix2pix\n'
    message += 'python train.py'
    raise Exception(message)

#############
## imports ##
#############

import time
from utils.training.discriminator import train_discriminator
from utils.training.generator     import train_generator

###############
## functions ##
###############

"""
    Name: train_one_epoch
    Description: Trains the discriminator and the generator one epoch.
    Inputs:
        >> train_elements: (TrainingElements) object.
        >> writer: (tensorboard SummaryWriter) object to save different values of the training process.
        >> step: (Int) Current step of the training. Needed for the SummaryWriter.
        >> test_after_n_epochs: (Int) Show info after n epochs.
    Outputs:
        >> step: (Int) Updated value for the step variable of the training. Needed for the SummaryWriter.
"""
def train_one_epoch(train_elements, writer, step, test_after_n_epochs):
    for batch_idx, (source_imgs, dst_imgs) in enumerate(train_elements.train_dataloader):
        batch_init_time = time.perf_counter()
        device = train_elements.device

        # Data to device and to proper data type
        source_imgs = source_imgs.to(device)
        dst_imgs = dst_imgs.to(device)

        ## Train discriminator: max log(D(x)) + log(1-D(G(z)))
        loss_disc, d_x, d_gx1 = train_discriminator(train_elements, source_imgs, dst_imgs)

        ## Train generator: min log(1-D(G(z))) <--> max log(D(G(z)))
        loss_gen, d_gx2 = train_generator(train_elements, source_imgs, dst_imgs)

        batch_final_time = time.perf_counter()
        batch_exec_time = batch_final_time - batch_init_time
        
        if batch_idx % test_after_n_epochs == 0 and batch_idx !=0:
            print(f'Epoch {train_elements.epoch}/{train_elements.hyperparams.total_epochs} - Batch {batch_idx}/{train_elements.total_train_baches} - Loss D {loss_disc:.6f} - Loss G {loss_gen:.6f} - D(x): {d_x:.6f} - D(G(x))_1: {d_gx1:.6f} - D(G(x))_2: {d_gx2:.6f} - Batch time {batch_exec_time:.6f} s.')
            writer.add_scalars( f'Loss/', {'Gen': loss_gen, 'Disc': loss_disc}, step)
            writer.add_scalars( f'Disc val/', {'D(x)': d_x, 'D(G(x))_1': d_gx1, 'D(G(x))_2': d_gx2}, step)
            step=step+1
    return step
