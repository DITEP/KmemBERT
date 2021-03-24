import torch.nn as nn
import torch
from torch.optim import Adam

from .health_bert import HealthBERT

class MultiEhrModel(nn.Module):
    def __init__(self, device, config, hidden_size_gru=20):
        super(MultiEhrModel, self).__init__()

        self.device = device
        self.health_bert = HealthBERT(device, config)
        for param in self.health_bert.parameters():
            param.requires_grad = False

        self.GRU = nn.Sequential(nn.GRU(input_size  = 2, hidden_size = hidden_size_gru, batch_first=True),
        nn.Linear(hidden_size_gru,1),
        nn.Sigmoid())

        self.optimizer = Adam(self.GRU.parameters(), lr = config.learning_rate, weight_decay=config.weight_decay)
        self.MSELoss = nn.MSELoss()


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
        seq = self.HealthBERT.step(texts).view(1,-1)
        seq = torch.cat(seq, dt, dim=1)
        out = self.GRU(seq)
        return out

    
        