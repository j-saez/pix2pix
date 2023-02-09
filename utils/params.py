#############
## imports ##
#############

import os
import argparse
from dataclasses import dataclass

#############
## classes ##
#############

"""
    DatasetParams
    Class description: Contains the hyperparams params for the training.
"""
@dataclass
class Hyperparams():
    # Class for storing the hyperparameter values
    total_epochs: int 
    test_after_n_epochs: int 
    batch_size: int 
    lr: float 
    adam_beta1: float
    adam_beta2: float
    patch_size: int
    use_unet_gen: bool
    l1_lambda: float

"""
    DatasetParams
    Class description: Contains the dataset params for the training.
"""
@dataclass
class DatasetParams():
    # Class for storing the dataset information
    dataset_path: str
    dataset_name: str
    img_size: int
    direction: str

"""
    Params
    Class description: Class to load the terminal arguments for the training,
                       containing the hyperparams and also the params for the dataset.
    Inputs: None
"""
class Params():
    def __init__(self):
        self.args = self.get_args()
        return

    """
        get_params
        Description: Returns the dataset params and the hyperparams for the training loop.
        Inputs: None
        Outputs:
            >> hyperparams: Hyperparams for the training loop.
            >> dataset_params: Params for the dataset.
    """
    def get_params(self):
        return self.get_hyperparams(), self.get_dataset_params()

    """
        get_hyperparams
        Description: Returns a Hyperparams object containing the hyperparams for the training loop.
        Inputs: 
            >> verbose: Wheter to show or not the params for the dataset.
        Outputs:
            >> hyperparams: Hyperparams for the training loop.
    """
    def get_hyperparams(self, verbose=True):
        hyperparams = Hyperparams(
            total_epochs=self.args.total_epochs,
            test_after_n_epochs=self.args.test_after_n_epochs,
            batch_size=self.args.batch_size,
            lr=self.args.learning_rate,
            adam_beta1=self.args.adam_beta1,
            adam_beta2=self.args.adam_beta2,
            l1_lambda=self.args.l1_lambda,
            patch_size=self.args.patch_size,
            use_unet_gen=self.args.use_unet_gen,
        )
        if verbose:
            print('\tHyperparams values.')
            for field in hyperparams.__dataclass_fields__:
                value = getattr(hyperparams, field)
                print(f'\t\t{field} = {value}')
        return hyperparams

    """
        get_dataset_params
        Description: returns a DatasetParams object with the params for the dataset.
        Inputs: 
            >> verbose: Wheter to show or not the params for the dataset.
        Outputs:
            >> dataset_params: Params for the dataset.
    """
    def get_dataset_params(self, verbose=True):

        dataset_params = DatasetParams(
            dataset_path=self.args.dataset_path,
            dataset_name=self.args.dataset_name,
            img_size=self.args.img_size,
            direction=self.args.direction,
        )
        if verbose:
            print('\tDataset params')
            for field in dataset_params.__dataclass_fields__:
                value = getattr(dataset_params, field)
                print(f'\t\t{field} = {value}')
        return dataset_params


    """
        get_args
        Description: Returns the arguments for the training loop and dataset.
        Inputs: None
        Outputs:
            >> args: Arguments loaded by the argument parser.
    """
    def get_args(self):
        parser = argparse.ArgumentParser(description='Arguments for pix2pix training.')

        # Hyperparams
        parser.add_argument( '--total_epochs', type=int, default=500, help='Total epochs for the training.')
        parser.add_argument( '--batch_size', type=int, default=16, help='Batch size for the training phase.')
        parser.add_argument( '--learning_rate', type=float, default=2e-4, help='Learning rate value.')
        parser.add_argument( '--adam_beta1', type=float, default=0.5,help='Adam beta 1.')
        parser.add_argument( '--adam_beta2', type=float, default=0.999,help='Adam beta 2.')
        parser.add_argument( '--l1_lambda',  type=float, default=100.0,help='L1 lambda. (Check eq 4 from original paper.)')
        parser.add_argument( '--test_after_n_epochs', type=int, default=10, help='Test the model after n epochs of training')
        parser.add_argument( '--patch_size', type=int, default=286, help='Patch for the discriminator. Select between 1, 16, 70 or 286 (default).')
        parser.add_argument( '--use_unet_gen', type=int, default=True, help='Whether to use the UNET (1) or the encoder-decoder generator (0).')
            
        # Dataset params
        parser.add_argument( '--dataset_path', type=str, default=os.getcwd()+'/datasets/', help='Path to the datasets folder.')
        parser.add_argument( '--dataset_name', type=str, default='maps', help='Name of the dataset to be loaded for the training.')
        parser.add_argument( '--img_size', type=int, default=256, help='Size for squared images.')
        parser.add_argument( '--direction', type=str, default="a_to_b", help='Direction for the transformation. Choose between a_to_b or b_to_a.')

        return parser.parse_args()
