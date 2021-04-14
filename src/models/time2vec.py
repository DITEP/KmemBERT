import torch
from torch import nn

class Time2Vec(nn.Module):
    """
    Transforms time into a vector
    This vector is used as a positional encoding for the transformer_aggregator

    Please see https://arxiv.org/pdf/1907.05321.pdf
    """
    def __init__(self, d_model):
        super(Time2Vec, self).__init__()

        self.d_model = d_model
        self.linear = nn.Linear(1, self.d_model)

    def forward(self, t):
        """
        Computes the positional encoding for a tensor of times
        For a given time, its corresponding positional encoding
        is a vector of size 768.

        Args:
            t (tensor of Nx1 times): times list

        Output:
            positional encoding (tensor of size Nx768)
        """
        x = self.linear(t)
        return torch.cat((x[:1], torch.sin(x[1:])))