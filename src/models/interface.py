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
    def __init__(self, device, config):
        super(ModelInterface, self).__init__()
        self.device = device
        self.config = config
        self.best_loss = np.inf
        self.early_stopping = 0
        self.start_epoch_timers()

    def start_epoch_timers(self):
        self.encoding_time = 0
        self.compute_time = 0

    def initialize_scheduler(self, epochs, train_loader):
        total_steps = len(train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                        num_warmup_steps=2, # Default value
                                                        num_training_steps=total_steps)

    def resume(self, config):
        printc(f"Resuming with model at {config.resume}...", "INFO")
        path_checkpoint = os.path.join(os.path.dirname(config.path_result), config.resume, 'checkpoint.pth')
        assert os.path.isfile(path_checkpoint), 'Error: no checkpoint found!'
        checkpoint = torch.load(path_checkpoint, map_location=self.device)
        
        self.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError
    
        