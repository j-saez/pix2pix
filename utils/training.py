import os
if os.getcwd()[-6:] != 'dc_gan':
    message = 'Run the file from the the root dir:\n'
    message += 'cd dc_gan\n'
    message += 'python train.py'
    raise Exception(message)

import time
import torch
from torch.utils.tensorboard import SummaryWriter

def load_tensorboard_writer(hyperparams, dataset_name, norm):
    tensorboard_dir = os.getcwd()+'/runs/tensorboard/'
    weights_dir = os.getcwd()+'/runs/weights/'
    model_logs_dir = os.getcwd()+f'/runs/tensorboard/{dataset_name}_lr{hyperparams.lr}_e{hyperparams.total_epochs}_zDim{hyperparams.z_dim}_layersNorm{norm}/'
    model_weights_dir = os.getcwd()+f'/runs/weights/{dataset_name}_lr{hyperparams.lr}_e{hyperparams.total_epochs}_zDim{hyperparams.z_dim}_layersNorm{norm}/'

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

def train_discriminator(disc, criterion, disc_opt, real_imgs, dst_imgs, fake_dst_imgs):
    disc.zero_grad()
    ## Train discriminator: max log(D(x)) + log(1-D(G(z)))

    real_disc_output = disc(real_imgs, dst_imgs).reshape(-1)
    loss_real_disc = criterion(real_disc_output, torch.ones_like(real_disc_output))
    loss_real_disc.backward()

    fake_disc_output = disc(real_imgs, fake_dst_imgs.detach()).reshape(-1)
    loss_fake_disc = criterion(fake_disc_output, torch.zeros_like(fake_disc_output))
    loss_fake_disc.backward()

    # From DCGAN
    # It is needed to add something more to the loss as described in the paper.
    loss_disc = ((loss_real_disc + loss_fake_disc) / 2.0) + t
    disc_opt.step()
    return loss_disc, real_disc_output.mean().item(), fake_disc_output.mean().item()

def train_generator(disc,gen,criterion,gen_opt,fake_imgs):
    gen.zero_grad()
    ## Train generator: min log(1-D(G(z))) <--> max log(D(G(z)))
    fake_disc_output = disc(fake_imgs).reshape(-1)
    loss_generator = criterion(fake_disc_output, torch.ones_like(fake_disc_output))
    loss_generator.backward()
    gen_opt.step()
    return loss_generator, fake_disc_output.mean().item()

def train_one_epoch(train_dataloader,disc,gen,disc_opt,gen_opt,criterion,hyperparms,epoch,total_train_baches,device,writer,step):
    for batch_idx, (orig_imgs, dst_imgs) in enumerate(train_dataloader):
        batch_init_time = time.perf_counter()

        # Data to device and to proper data type
        real_imgs = orig_imgs.to(device)
        dst_imgs = dst_imgs.to(device)
        fake_dst_imgs = gen(orig_imgs)

        ## Train discriminator: max log(D(x)) + log(1-D(G(z)))
        loss_disc, d_x, d_gx1 = train_discriminator(disc, criterion, disc_opt, real_imgs, dst_imgs, fake_dst_imgs)

        ## Train generator: min log(1-D(G(z))) <--> max log(D(G(z)))
        loss_gen, d_gx2 = train_generator(disc, gen, criterion, gen_opt, fake_dst_imgs)

        batch_final_time = time.perf_counter()
        batch_exec_time = batch_final_time - batch_init_time
        
        if batch_idx % hyperparms.test_after_n_epochs == 0 and batch_idx !=0:
            # To be honest, in GANs the loss does not say much 
            print(f'Epoch {epoch}/{hyperparms.total_epochs} - Batch {batch_idx}/{total_train_baches} - Loss D {loss_disc:.6f} - Loss G {loss_gen:.6f} - D(x): {d_x:.6f} - D(G(x))_1: {d_gx1:.6f} - D(G(x))_2: {d_gx2:.6f} - Batch time {batch_exec_time:.6f} s.')
            writer.add_scalars( f'Loss/', {'Gen': loss_gen, 'Disc': loss_disc}, step)
            writer.add_scalars( f'Disc val/', {'D(x)': d_x, 'D(G(x))_1': d_gx1, 'D(G(x))_2': d_gx2}, step)
            step=step+1
    return step

