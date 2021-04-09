'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import torch.nn as nn
import torch
from torch.optim import Adam

from .interface import ModelInterface
from .time2vec import Time2Vec

class TransformerAggregator(ModelInterface):
    mode = 'multi'

    def __init__(self, device, config, d_model, nhead, num_layers, out_dim):
        super(TransformerAggregator, self).__init__(device, config)

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.out_dim = out_dim

        assert self.out_dim in [1, 2], f'TransformersMulti out_dim should be 1 or 2. Found {out_dim}.'

        self.t2v = Time2Vec(self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.out_proj = nn.Linear(self.d_model, self.out_dim)

        self.optimizer = Adam(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.MSELoss = nn.MSELoss()

        self.to(self.device)

    def step(self, *inputs):
        outputs, dt, label = inputs

        outputs = outputs[0].to(self.device)
        dt = dt.to(self.device)
        label = label.to(self.device)

        output = self(outputs, dt)

        if self.out_dim == 2:
            mu, log_var = output[0], output[1]
            mu = torch.sigmoid(mu)
            loss = log_var + (label - mu)**2/torch.exp(log_var)
            output = (mu, log_var)
        else:
            mu = torch.sigmoid(output)
            loss = self.MSELoss(mu, label)
            output = mu

        if self.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, output

    def forward(self, outputs, dt):
        positional_encoding = self.t2v(dt[:, None])
        x = outputs + positional_encoding
        x = self.transformer_encoder(x[None, :])
        x = self.out_proj(x[0, -1])
        return x
