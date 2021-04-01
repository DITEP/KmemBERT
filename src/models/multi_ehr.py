'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import torch.nn as nn
import torch
from torch.optim import Adam
import json
import os

from .interface import ModelInterface
from .health_bert import HealthBERT
from ..utils import printc, shift_predictions

class MultiEHR(ModelInterface):
    """
    Runs a GRU on Health Bert predictions on multiple EHRs
    """

    mode = 'multi'

    def __init__(self, device, config, hidden_size_gru=16, nb_gru_layers=1):
        super(MultiEHR, self).__init__(device, config)

        self.nb_gru_layers = nb_gru_layers
        self.hidden_size_gru = hidden_size_gru

        if self.config.mode == 'regression':
            self.input_size = 2
        elif self.config.mode == 'density':
            self.input_size = 3
        else:
            raise ValueError(f"Invalid health bert mode. Needed 'regression' or 'density', found {self.config.mode}")

        self.GRU = nn.GRU(input_size=self.input_size, num_layers=self.nb_gru_layers, hidden_size=hidden_size_gru, batch_first=True)

        self.out_proj = nn.Linear(hidden_size_gru, self.input_size-1)

        self.optimizer = Adam(self.GRU.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.MSELoss = nn.MSELoss()
        
        self.eval()

    def init_hidden(self):
        hidden = torch.empty(1, self.nb_gru_layers, self.hidden_size_gru)
        return nn.init.xavier_uniform_(hidden, gain=nn.init.calculate_gain('relu')).to(self.device)

    def step(self, *inputs):
        """
        Args:
            outputs (tuple of tensors): health bert predictions - either (mus,) or (mus, log_vars)
            dt (1D tensor): list of timestamps between the creation of EHRs and the last EHR of the sequence
            label (one element tensor): label for the given list of EHRs

        Outputs:
            loss
            output: mu on regression mode or (mu, log_var) on density mode

        Examples::
            >>> outputs = [torch.tensor([0.7, 0.2, 0.3]), torch.tensor([-2.7, 0.8, -11])]
            >>> dt = torch.tensor([230, 150, 0]) # Number of days
            >>> label = torch.tensor(0.27)
            >>> model.step(outputs, dt, label)
        """
        outputs, dt, label = inputs

        outputs = [output.to(self.device) for output in outputs]
        dt = dt.to(self.device)
        label = label.to(self.device)

        if self.training:
            self.optimizer.zero_grad()

        output = self(outputs, dt)
        if self.config.mode == 'density':
            output = torch.split(output, 1)
            mu, log_var = output
            mu = torch.sigmoid(mu)
            loss = log_var + (label - mu)**2/torch.exp(log_var)
            output = (mu, log_var)
        else:
            mu = torch.sigmoid(output)
            loss = self.MSELoss(mu, label)
            output = mu

        if self.training:
            loss.backward()
            self.optimizer.step()

        return loss, output

   
    def forward(self, outputs, dt):
        hidden = self.init_hidden()

        seq = torch.stack((*outputs, dt)).T

        out = self.GRU(seq[None,:], hidden)[1]
        out = self.out_proj(out).view(-1)
        return out

    def train(self):
        self.GRU.train()

    def eval(self):
        self.GRU.eval()
    

class Conflation(ModelInterface):
    """
    Predicts survival times based on a conflation method to aggregate multiple predictions
    """

    mode = 'multi'

    def __init__(self, device, config):
        super(Conflation, self).__init__(device, config)

    def step(self, *inputs):
        outputs, dt, _ = inputs

        outputs = [output.to(self.device) for output in outputs]
        dt = dt.to(self.device)

        if self.config.mode == 'density':
            mus, log_vars = outputs
        else:
            mus, *_ = outputs
            log_vars = dt / self.config.mean_time_survival

        mus = shift_predictions(mus, self.config.mean_time_survival, dt)
        return torch.zeros(1), self(mus, log_vars)


    def forward(self, mus, log_vars):
        """
        Conflation of gaussian densities

        Args:
            mus (tensor): tensor of means
            log_vars (tensor): tensor of logs of variances

        Outputs:
            mu or (mu, log_var)
        """
        vars = torch.exp(log_vars)
        vars_product = vars.prod()
        ones = torch.ones(len(vars)).to(self.device)
        products = vars_product*ones / vars

        mu = torch.dot(mus, products) / products.sum()
        if self.config.mode == 'density':
            return (mu, torch.log(vars_product / products.sum()))
        else:
            return mu


    def train(self, *args):
        pass

    def eval(self, *args):
        pass

class HealthCheck(ModelInterface):
    """
    Does nothing except returning Health Bert predictions 
    It should have the same score than the loaded checkpoint
    """
    mode = 'multi'

    def __init__(self, device, config):
        super(HealthCheck, self).__init__(device, config)

    def step(self, *inputs):
        outputs, dt, _ = inputs

        if self.config.mode == 'density':
            mus, log_vars = outputs
            return torch.zeros(1), (mus[-1], log_vars[-1])
        else:
            mus, *_ = outputs
            return torch.zeros(1), mus[-1]

    def train(self, *args):
        pass

    def eval(self, *args):
        pass