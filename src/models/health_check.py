import torch

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