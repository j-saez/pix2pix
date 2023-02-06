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
import torch
from torch.utils.tensorboard import SummaryWriter

###############
## functions ##
###############

"""
    Name: load_tensorboard_writer
    Description: Loads the tensorboard writer object.
    Inputs:
        >> hyperparms: Hyperparms (check <root repo>/utils/params.py) object containing the value of the diffrerent hypervalues containing the value of the diffrerent hypervalues.
        >> dataset_name: (str) Name of the dataset the model is being trained with.
    Outputs:
        >> writer: tensorboard.SummaryWriter to save the training data.
        >> model_weights_dir: (str) containing the path where the weights of the models will be saved.
"""
def load_tensorboard_writer(hyperparams, dataset_name):
    tensorboard_dir = os.getcwd()+'/runs/tensorboard/'
    weights_dir = os.getcwd()+'/runs/weights/'
    model_logs_dir = os.getcwd()+f'/runs/tensorboard/{dataset_name}_lr{hyperparams.lr}_e{hyperparams.total_epochs}_/'
    model_weights_dir = os.getcwd()+f'/runs/weights/{dataset_name}_lr{hyperparams.lr}_e{hyperparams.total_epochs}_/'

    if not os.path.isdir(os.getcwd()+'/runs'):
        os.mkdir(os.getcwd()+'/runs')
    if not os.path.isdir(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    if not os.path.isdir(weights_dir):
        os.mkdir(weights_dir)
    if not os.path.isdir(model_logs_dir):
        os.mkdir(model_logs_dir)
    if not os.path.isdir(model_weights_dir):
        os.mkdir(model_weights_dir)

    writer = SummaryWriter(log_dir=model_logs_dir)
    return writer, model_weights_dir

"""
    Name: train_discriminator
    Description: Trains the discriminator one epoch.
    Inputs:
        >> disc: (nn.Module) Discriminator model.
        >> bce_criterion: (nn.BCELoss) BCELoss object.
        >> disc_opt: (nn.optim) Discriminator optimizer.
        >> real_imgs: (torch tensor) containing the real images from the source domain.
        >> dst_imgs: (torch tensor) containing the real images from the target domain.
        >> fake_dst_imgs: (torch tensor) containing the fake iamges from the target domain.
    Outputs:
        >> loss_disc: (torch.float32) containing the value of the loss for the discriminator
        >> real_disc_output: (torch.float32) containing the value of the output for the discriminator when fed with real images.
        >> fake_disc_output: (torch.float32) containing the value of the output for the discriminator when fed with fake images.
"""
def train_discriminator(disc, bce_criterion, disc_opt, real_imgs, dst_imgs, fake_dst_imgs):
    disc.zero_grad()

    ## Train discriminator: max log(D(x)) + log(1-D(G(z)))
    real_disc_output = disc(real_imgs, dst_imgs).reshape(-1)
    loss_real_disc = bce_criterion(real_disc_output, torch.ones_like(real_disc_output))
    loss_real_disc.backward()

    fake_disc_output = disc(real_imgs, fake_dst_imgs.detach()).reshape(-1)
    loss_fake_disc = bce_criterion(fake_disc_output, torch.zeros_like(fake_disc_output))
    loss_fake_disc.backward()

    # It is needed to add something more to the loss as described in the paper.
    loss_disc = ((loss_real_disc + loss_fake_disc) / 2.0) 
    disc_opt.step()
    return loss_disc, real_disc_output.mean().item(), fake_disc_output.mean().item()

"""
    Name: train_generator
    Description: Trains the generator one epoch.
    Inputs:
        >> disc: (nn.Module) Discriminator model.
        >> gen: (nn.Module) Generator model.
        >> bce_criterion: (nn.BCELoss) BCELoss object.
        >> l1_criterion: (nn.L1Loss) L1Loss object.
        >> l1_lambda: (float) containing the value for the labmda that multiplies the l1 loss value.
        >> gen_opt: (nn.optim) Generator optimizer.
        >> fake_imgs: (torch tensor) containing the fake iamges from the target domain.
    Outputs:
        >> loss_generator: (torch.float32) containing the value of the loss for the generator.
        >> fake_disc_output: (torch.float32) containing the value of the output for the discriminator when fed with fake images.
"""
def train_generator(disc,gen,bce_criterion,l1_criterion,l1_lambda,gen_opt,real_imgs,fake_dst_imgs):
    gen.zero_grad()
    ## Train generator: min log(1-D(G(z))) <--> max log(D(G(z)))
    fake_disc_output = disc(real_imgs, fake_dst_imgs).reshape(-1)
    bce_loss_generator = bce_criterion(fake_disc_output, torch.ones_like(fake_disc_output))
    l1_loss_generator = l1_criterion(fake_disc_output, torch.ones_like(fake_disc_output))
    loss_generator = bce_loss_generator + l1_lambda * l1_loss_generator
    loss_generator.backward()
    gen_opt.step()
    return loss_generator, fake_disc_output.mean().item()

"""
    Name: train_one_epoch
    Description: Trains the discriminator and the generator one epoch.
    Inputs:
        >> train_dataloader: Dataloader for the training dataset.
        >> disc: (nn.Module) Discriminator model.
        >> gen: (nn.Module) Generator model.
        >> disc_opt: (nn.optim) Discriminator optimizer.
        >> gen_opt: (nn.optim) Generator optimizer.
        >> bce_criterion: (nn.BCELoss) object containing the bce criterion.
        >> l1_criterion: (torch.nn.) object containing the l1 criterion.
        >> hyperparms: Hyperparms (check <root repo>/utils/params.py) object containing the value of the diffrerent hypervalues containing the value of the diffrerent hypervalues.
        >> epoch: (Int) Current epoch of the training.
        >> total_train_baches: (Int) Total number of baches for the training dataset.
        >> device: Device where the data is going to be loaded for training.
        >> writer: (tensorboard SummaryWriter) object to save different values of the training process.
        >> step: (Int) Current step of the training. Needed for the SummaryWriter.
    Outputs:
        >> step: (Int) Updated value for the step variable of the training. Needed for the SummaryWriter.
"""
def train_one_epoch(train_dataloader,disc,gen,disc_opt,gen_opt,bce_criterion,l1_criterion,hyperparms,epoch,total_train_baches,device,writer,step):
    for batch_idx, (real_imgs, dst_imgs) in enumerate(train_dataloader):
        batch_init_time = time.perf_counter()

        # Data to device and to proper data type
        real_imgs = real_imgs.to(device)
        dst_imgs = dst_imgs.to(device)
        fake_dst_imgs = gen(real_imgs)

        ## Train discriminator: max log(D(x)) + log(1-D(G(z)))
        loss_disc, d_x, d_gx1 = train_discriminator(disc, bce_criterion, disc_opt, real_imgs, dst_imgs, fake_dst_imgs)

        ## Train generator: min log(1-D(G(z))) <--> max log(D(G(z)))
        loss_gen, d_gx2 = train_generator(disc, gen, bce_criterion, l1_criterion, hyperparms.l1_lambda, gen_opt, real_imgs, fake_dst_imgs)

        batch_final_time = time.perf_counter()
        batch_exec_time = batch_final_time - batch_init_time
        
        if batch_idx % hyperparms.test_after_n_epochs == 0 and batch_idx !=0:
            # To be honest, in GANs the loss does not say much 
            print(f'Epoch {epoch}/{hyperparms.total_epochs} - Batch {batch_idx}/{total_train_baches} - Loss D {loss_disc:.6f} - Loss G {loss_gen:.6f} - D(x): {d_x:.6f} - D(G(x))_1: {d_gx1:.6f} - D(G(x))_2: {d_gx2:.6f} - Batch time {batch_exec_time:.6f} s.')
            writer.add_scalars( f'Loss/', {'Gen': loss_gen, 'Disc': loss_disc}, step)
            writer.add_scalars( f'Disc val/', {'D(x)': d_x, 'D(G(x))_1': d_gx1, 'D(G(x))_2': d_gx2}, step)
            step=step+1
    return step
