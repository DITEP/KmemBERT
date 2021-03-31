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

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size_gru, 1),
            nn.Sigmoid()
        )

        self.optimizer = Adam(self.GRU.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.MSELoss = nn.MSELoss()
        
        self.eval()

    def init_hidden(self):
        hidden = torch.empty(1, self.nb_gru_layers, self.hidden_size_gru)
        return nn.init.xavier_uniform_(hidden, gain=nn.init.calculate_gain('relu')).to(self.device)

    def step(self, *inputs):
        *outputs, dt, label = inputs
        dt = dt.to(self.device)
        label = label.to(self.device)
        if self.training:
            self.optimizer.zero_grad()

        output = self(outputs, dt)
        loss = self.MSELoss(output, label)

        if self.training:
            loss.backward()
            self.optimizer.step()

        return loss, output

   
    def forward(self, outputs, dt):
        hidden = self.init_hidden()

        if self.config.mode == 'density':
            seq = torch.cat((*outputs, dt)).T
        else:
            seq = outputs[0].view(-1)
            seq = torch.stack((seq, dt[0]), dim=1)

        out = self.GRU(seq[None,:], hidden)[1]
        out = self.out_proj(out).view(-1)
        return out

    def train(self):
        self.GRU.train()

    def eval(self):
        self.GRU.eval()
    

class Conflation(ModelInterface):
    mode = 'multi'

    def __init__(self, device, config):
        super(Conflation, self).__init__(device, config)

    def step(self, *inputs):
        *outputs, dt, _ = inputs
        dt = dt.to(self.device)[0]

        if self.config.mode == 'density':
            mus, log_vars = outputs
            mus, log_vars = mus[0], log_vars[0]
        
        else:
            mus, *_ = outputs
            mus = mus.view(-1)
            log_vars = dt / self.config.mean_time_survival

        mus = shift_predictions(mus, self.config.mean_time_survival, dt)
        return torch.zeros(1), self(mus, log_vars)


    def forward(self, mus, log_vars):
        vars = torch.exp(log_vars)
        vars_product = vars.prod()
        products = vars_product*torch.ones(len(vars)) / vars
        return torch.dot(mus, products) / products.sum() #, torch.log(vars_product / products.sum())

    def train(self, *args):
        pass

    def eval(self, *args):
        pass

class HealthCheck(ModelInterface):
    mode = 'multi'

    def __init__(self, device, config):
        super(HealthCheck, self).__init__(device, config)

    def step(self, *inputs):
        *outputs, dt, _ = inputs

        mus, *_ = outputs
        mus = mus.view(-1)

        return torch.zeros(1), mus[-1]

    def train(self, *args):
        pass

    def eval(self, *args):
        pass