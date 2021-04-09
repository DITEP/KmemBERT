import torch
from torch import nn
from torch.optim import Adam

from .interface import ModelInterface

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

class HealthCheckTransformer(ModelInterface):
    mode = 'multi'
    out_dim = 1

    def __init__(self, device, config):
        super(HealthCheckTransformer, self).__init__(device, config)

        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(768, 1)

        self.optimizer = Adam(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.MSELoss = nn.MSELoss()

        self.to(self.device)

    def step(self, *inputs):
        outputs, dt, label = inputs

        outputs = outputs[0].to(self.device)
        dt = dt.to(self.device)
        label = label.to(self.device)

        output = self(outputs, dt)

        mu = torch.sigmoid(output)
        loss = self.MSELoss(mu, label)
        output = mu

        if self.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, output

    def forward(self, outputs, _):
        x = outputs[-1]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
