'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import numpy as np
import torch.nn as nn
import torch
from transformers import get_linear_schedule_with_warmup
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from ..utils import printc

class ModelInterface(nn.Module):
    """
    PyTorch Model interface
    Every model has to inherit from ModelInterface so training and testing run correctly

    At least the following propeties / method has to be implemented, the others can be overridden
    - self.optimizer
    - the methods defined below and which raise NotImplementedError
    """
    def __init__(self, device, config):
        super(ModelInterface, self).__init__()
        self.device = device
        self.config = config
        self.best_loss = np.inf
        self.early_stopping = 0

    def initialize_scheduler(self, epochs=0, train_loader=[]):
        """
        Creates a scheduler for a given otimizer
        """
        total_steps = len(train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                        num_warmup_steps=2, # Default value
                                                        num_training_steps=total_steps)

    def resume(self, config):
        """
        Resumes with a given checkpoint. Loads the saved parameters, optimizer and scheduler.
        """
        printc(f"Resuming with model at {config.resume}...", "INFO")
        path_checkpoint = os.path.join(os.path.dirname(config.path_result), config.resume, 'checkpoint.pth')
        assert os.path.isfile(path_checkpoint), 'Error: no checkpoint found!'
        checkpoint = torch.load(path_checkpoint, map_location=self.device)
        
        self.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

    def step(self, *args, **kwargs):
        """
        Args:
            *args, **kwargs: data from the data loaders (see training.py)

        Output:
            loss (tensor): PyTorch loss
            outputs: model outputs (predictions or something else)
        
        Examples::
            >>> *data, labels = next(iter(loader))
            >>> loss, outputs = model.step(*data, labels)
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """
        PyTorch nn.Module forward
        It is specific to the model, and the args have no specific format
        """
        raise NotImplementedError
    
        