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
from ..utils import printc

class MultiEHR(ModelInterface):
    mode = 'multi'

    def __init__(self, device, config, hidden_size_gru=16, nb_gru_layers=1):
        super(MultiEHR, self).__init__(device, config)

        with open(os.path.join('results', config.resume, 'args.json')) as json_file:
            self.health_bert_mode = json.load(json_file)["mode"]
            assert self.health_bert_mode != "classify", "Health Bert mode classify not supported for RNNs"
            printc(f"\nUsing Health BERT checkpoint {config.resume} using mode {self.health_bert_mode}", "INFO")
            config.mode = self.health_bert_mode

        self.health_bert = HealthBERT(device, config)
        for param in self.health_bert.parameters():
            param.requires_grad = False

        self.nb_gru_layers = nb_gru_layers
        self.hidden_size_gru = hidden_size_gru

        if self.health_bert.mode == 'regression':
            self.input_size = 2
        elif self.health_bert.mode == 'density':
            self.input_size = 3
        else:
            raise ValueError(f"Invalid health bert mode. Needed 'regression' or 'density', found {self.health_bert.mode}")

        self.GRU = nn.GRU(input_size=self.input_size, num_layers=self.nb_gru_layers, hidden_size=hidden_size_gru, batch_first=True)

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size_gru, 1),
            nn.Sigmoid()
        )

        self.optimizer = Adam(self.GRU.parameters(), lr = config.learning_rate, weight_decay=config.weight_decay)
        self.MSELoss = nn.MSELoss()
        
        self.eval()

    def init_hidden(self):
        hidden = torch.empty(1, self.nb_gru_layers, self.hidden_size_gru)
        return nn.init.xavier_uniform_(hidden, gain=nn.init.calculate_gain('relu')).to(self.device)

    def step(self, texts, dt, label):
        dt = dt.to(self.device)
        label = label.to(self.device)
        if self.training:
            self.optimizer.zero_grad()

        output = self(texts[0], dt[0])
        loss = self.MSELoss(output, label)

        if self.training:
            loss.backward()
            self.optimizer.step()

        return loss, output

   
    def forward(self, texts, dt):
        hidden = self.init_hidden()

        seq = self.health_bert.step(texts)
        if self.health_bert_mode == 'density':
            seq = torch.stack((*seq, dt)).T
        else:
            seq = seq.view(-1)
            seq = torch.stack((seq, dt), dim=1)

        out = self.GRU(seq[None,:], hidden)[1]
        out = self.out_proj(out).view(-1)
        return out

    def train(self):
        self.GRU.train()

    def eval(self):
        self.GRU.eval()
    
        