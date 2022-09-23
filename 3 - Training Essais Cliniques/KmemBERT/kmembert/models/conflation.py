'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import torch

from .interface import ModelInterface
from ..utils import shift_predictions

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
