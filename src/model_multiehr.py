import torch.nn as nn
import torch
from torch.optim import Adam

from .health_bert import HealthBERT

class MultiEHRModel(nn.Module):
    def __init__(self, device, config, hidden_size_gru=20):
        super(MultiEHRModel, self).__init__()

        self.device = device
        self.health_bert = HealthBERT(device, config)
        for param in self.health_bert.parameters():
            param.requires_grad = False

        self.nb_gru_layers = 1

        self.hidden_size_gru = hidden_size_gru
        self.GRU = nn.GRU(input_size=2, num_layers=self.nb_gru_layers, hidden_size=hidden_size_gru, batch_first=True)

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size_gru,1),
            nn.Sigmoid()
        )

        self.optimizer = Adam(self.GRU.parameters(), lr = config.learning_rate, weight_decay=config.weight_decay)
        self.MSELoss = nn.MSELoss()

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

        seq = self.health_bert.step(texts).view(-1)
        seq = torch.stack((seq, dt), dim=1)[None,:]
        out = self.GRU(seq, hidden)
        out = out[1]
        out = self.out_proj(out).view(-1)
        return out

    def train(self):
        self.GRU.train()

    def eval(self):
        self.GRU.eval()
    
        