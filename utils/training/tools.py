import os
import torch
import torchvision
from dataclasses             import dataclass
from utils.params            import Hyperparams
from torch.utils.tensorboard import SummaryWriter

#############
## classes ##
#############

@dataclass
class TrainModelsAndOpts:
    disc:     torch.nn.Module
    gen:      torch.nn.Module
    disc_opt: torch.optim.Optimizer
    gen_opt:  torch.optim.Optimizer

@dataclass
class TrainCriterions:
    bce: torch.nn.Module
    l1:  torch.nn.Module

@dataclass
class TrainingElements:
    train_dataloader:   torch.utils.data.DataLoader
    models:             TrainModelsAndOpts
    criterions:         TrainCriterions
    hyperparams:        Hyperparams
    epoch:              int
    total_train_baches: int
    device:             torch.device
    random_cropper:     torchvision.transforms.RandomCrop

###############
## functions ##
###############

"""
    create_runs_dirs
    Description: Creates the directories to save all the data in tensorboards and the models' weights.
    Inputs:
        >> tensorboard_dir
        >> weights_dir
        >> model_weights_dir
        >> model_logs_dir
    Outputs: None
"""
def create_runs_dirs(tensorboard_dir, weights_dir, model_weights_dir, model_logs_dir) -> None:
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
    return

"""
    Name: load_tensorboard_writer
    Description: Loads the tensorboard writer object.
    Inputs:
        >> hyperparms: hyperparams (check <root repo>/utils/params.py) object containing the value of the diffrerent hypervalues containing the value of the diffrerent hypervalues.
        >> dataset_name: (str) Name of the dataset the model is being trained with.
    Outputs:
        >> writer: tensorboard.SummaryWriter to save the training data.
        >> model_weights_dir: (str) containing the path where the weights of the models will be saved.
"""
def load_tensorboard_writer(hyperparams, dataset_name):
    selected_gen = 'unet' if hyperparams.use_unet_gen else 'encoderDecoder'
    tensorboard_dir = os.getcwd()+'/runs/tensorboard/'
    weights_dir = os.getcwd()+'/runs/weights/'
    training_dir_name = f'/runs/tensorboard/{dataset_name}_bs{hyperparams.batch_size}_lr{hyperparams.lr}_e{hyperparams.total_epochs}_{hyperparams.patch_size}Disc_{selected_gen}Gen/'
    model_logs_dir = os.getcwd() + training_dir_name
    model_weights_dir = os.getcwd() + training_dir_name

    create_runs_dirs(tensorboard_dir, weights_dir, model_weights_dir, model_logs_dir)
    writer = SummaryWriter(log_dir=model_logs_dir)
    return writer, model_weights_dir
