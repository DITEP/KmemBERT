import torch
from torch import nn

class Time2Vec(nn.Module):
    def __init__(self, d_model):
        super(Time2Vec, self).__init__()

        self.d_model = d_model
        self.linear = nn.Linear(1, self.d_model)

    def forward(self, t):
        x = self.linear(t)
        return torch.cat((x[:1], torch.sin(x[1:])))