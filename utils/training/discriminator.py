import torch

"""
    Name: train_discriminator
    Description: Trains the discriminator one epoch.
    Inputs:
        >> disc: (nn.Module) Discriminator model.
        >> bce_criterion: (nn.BCELoss) BCELoss object.
        >> disc_opt: (nn.optim) Discriminator optimizer.
        >> real_imgs: (torch tensor) containing the real images from the source domain.
        >> dst_imgs: (torch tensor) containing the real images from the target domain.
    Outputs:
        >> loss_disc: (torch.float32) containing the value of the loss for the discriminator
        >> real_disc_output: (torch.float32) containing the value of the output for the discriminator when fed with real images.
        >> fake_disc_output: (torch.float32) containing the value of the output for the discriminator when fed with fake images.
"""
def train_discriminator(train_elements, real_imgs, dst_imgs):
    rand_cropper = train_elements.random_cropper
    disc = train_elements.models.disc
    gen = train_elements.models.gen
    bce = train_elements.criterions.bce

    ## Train discriminator: max log(D(x)) + log(1-D(G(z)))
    real_disc_output = disc(rand_cropper(real_imgs), rand_cropper(dst_imgs)).reshape(-1)
    loss_real_disc = bce(real_disc_output, torch.ones_like(real_disc_output))

    fake_disc_output = disc(rand_cropper(real_imgs), rand_cropper( gen(dst_imgs) )).reshape(-1)
    loss_fake_disc = bce(fake_disc_output, torch.zeros_like(fake_disc_output))
    loss_disc = ((loss_real_disc + loss_fake_disc) / 2.0) 

    disc.zero_grad()
    loss_disc.backward()
    train_elements.models.disc_opt.step()
    return loss_disc, real_disc_output.mean().item(), fake_disc_output.mean().item()
