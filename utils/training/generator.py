import torch

"""
    Name: train_generator
    Description: Trains the generator one epoch.
    Inputs:
        >> train_elements: (TrainingElements) object.
        >> x_imgs: (torch.Tensor) containg the x images.
        >> y_imgs: (torch.Tensor) containg the y images.
    Outputs:
        >> loss_generator: (torch.float32) containing the value of the loss for the generator.
        >> fake_disc_output: (torch.float32) containing the value of the output for the discriminator when fed with fake images.
"""
def train_generator(train_elements, x_imgs, y_imgs):
    rand_cropper = train_elements.random_cropper
    disc = train_elements.models.disc
    gen = train_elements.models.gen
    bce = train_elements.criterions.bce
    l1 = train_elements.criterions.l1
    l1_lambda = train_elements.hyperparams.l1_lambda

    ## Train generator: min log(1-D(G(z))) <--> max log(D(G(z)))
    fake_disc_output = disc(rand_cropper(x_imgs), rand_cropper(gen(y_imgs))).reshape(-1)
    bce_loss_generator = bce(fake_disc_output, torch.ones_like(fake_disc_output))
    l1_loss_generator = l1(gen(y_imgs), y_imgs)
    loss_generator = bce_loss_generator + l1_lambda * l1_loss_generator

    gen.zero_grad()
    loss_generator.backward()
    train_elements.models.gen_opt.step()
    return loss_generator, fake_disc_output.mean().item()
