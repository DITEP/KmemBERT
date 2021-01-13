import json

from transformers import CamembertForSequenceClassification
from farm.modeling.tokenization import Tokenizer

import torch
import torch.nn as nn
from torch.optim import Adam

from time import time
class HealthBERT(nn.Module):
    def __init__(self, device, lr, voc_path=None, model_name="camembert-base", classify=False, freeze=False):
        super(HealthBERT, self).__init__()
        
        self.device = device
        self.lr = lr
        self.voc_path = voc_path
        self.model_name = model_name
        self.classify = classify

        if self.classify:
            self.num_labels = 2
        else:
            self.num_labels = 1
            self.MSELoss = nn.MSELoss()

        self.camembert = CamembertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels)
        self.camembert.to(self.device)
        self.optimizer = Adam(self.camembert.parameters(), lr=self.lr)

        if freeze:
            self.freeze()

        self.tokenizer = Tokenizer.load(self.model_name, lower_case=False, fast=True)

        if self.voc_path:
            self.add_tokens_from_path(self.voc_path)

    def start_epoch_timers(self):
        self.encoding_time = 0
        self.compute_time = 0

    def freeze(self):
        self.frozen = True
        for param in self.camembert.roberta.parameters():
            param.requires_grad = False

    def forward(self, *input, **kwargs):
        if self.classify:
            return self.camembert(*input, **kwargs)
        else:
            return torch.sigmoid(self.camembert(*input, **kwargs).logits)

    def get_loss(self, outputs, labels=None):
        if self.classify:
            return outputs.loss
        else:
            return self.MSELoss(outputs.reshape(-1), labels)

    def step(self, texts, labels):
        encoding_start_time = time()
        encoding = self.tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True)
        self.encoding_time += time()-encoding_start_time

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        if not self.classify:
            labels = labels.type(torch.FloatTensor)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()

        compute_start_time = time()
        outputs = self(input_ids, attention_mask=attention_mask, labels=labels)
        self.compute_time += time()-compute_start_time


        loss = self.get_loss(outputs, labels=labels)

        loss.backward()
        self.optimizer.step()

        return loss, outputs.logits if self.classify else outputs

    def add_tokens_from_path(self, voc_path):
        with open(voc_path) as json_file:
            voc_list = json.load(json_file)
            new_tokens = self.tokenizer.add_tokens([ token for (token, _) in voc_list ])
            print(f"Added {new_tokens} tokens to the tokenizer")

        self.camembert.resize_token_embeddings(len(self.tokenizer))

    def train(self):
        self.camembert.train()

    def eval(self):
        self.camembert.eval()

