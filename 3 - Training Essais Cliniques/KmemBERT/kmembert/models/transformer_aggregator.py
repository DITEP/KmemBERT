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
    camembert_d_model = 768

    def __init__(self, device, config, nhead, num_layers, out_dim, time_dim):
        super(TransformerAggregator, self).__init__(device, config)

        self.d_model = self.camembert_d_model + time_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.out_dim = out_dim

        assert self.out_dim in [1, 2], f'TransformerAggregator out_dim should be 1 or 2. Found {out_dim}.'

        self.t2v = Time2Vec(time_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

        self.out_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, self.out_dim)
        )

        # Penser a modifier le learning_rate du optimizer
        self.optimizer = Adam(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.MSELoss = nn.MSELoss()
        self.BCELoss = nn.BCELoss()

        self.to(self.device)


    def step(self, *inputs):
        """
        Runs a transformer on multiple medical reports corresponding to one patient
        The positional encoding is based on Time2Vec (see time2vec.py)

        Args:
            outputs (tensor of size Nx768): health bert encoder outputs
            dt (tensor of size N): times diff with most recent medical report
            label (tensor): survival time in [0,1]

        Outputs:
            loss
            output: mu or (mu, log_var)
        """
        outputs, dt, label = inputs

        # Set input, dt and label to device (GPU)
        outputs = outputs[0].to(self.device)
        dt = dt.to(self.device)
        label = label.to(self.device)

        # Compute prediction
        output = self(outputs, dt)

        # Prediction to Score, then Loss
        mu, log_var = output[0], output[1]
        mu = torch.sigmoid(mu)

        loss = self.BCELoss(mu, label)
        output = (mu, log_var)

        # BackWard if training
        if self.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, output

    def forward(self, outputs, dt):

        #print(dt[:, None])

        time_encoding = self.t2v(dt[:, None])

        #print(time_encoding)

        x = torch.cat([outputs, time_encoding], dim=1)

        x = self.transformer_encoder(x[:, None, :])

        x = self.out_proj(x[-1, 0])

        return x
