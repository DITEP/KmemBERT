'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import json
import numpy as np
import os
from transformers import CamembertForSequenceClassification, CamembertTokenizerFast
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from time import time
logging.getLogger("transformers").setLevel(logging.ERROR)

from ..utils import printc
from .interface import ModelInterface

def set_dropout(model, drop_rate=0.1):
    for _, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)

class HealthBERT(ModelInterface):
    """
    Model that instanciates a camembert model, a tokenizer and an optimizer.
    It supports methods to train it.
    """
    def __init__(self, device, config):
        super(HealthBERT, self).__init__(device, config)

        self.learning_rate = config.learning_rate
        self.voc_path = config.voc_path
        self.model_name = config.model_name

        if config.mode is None:
            assert self.config.resume is not None, 'Mode was not specified, cannot init HealthBERT'
            with open(os.path.join('results', self.config.resume, 'args.json')) as json_file:
                self.config.mode = json.load(json_file)["mode"]
                printc(f"\nUsing mode {self.config.mode} (Health BERT checkpoint {self.config.resume})", "INFO")
        self.mode = self.config.mode

        if self.mode == 'classif' or self.mode == 'density':
            self.num_labels = 2
        else:
            self.num_labels = 1
            self.MSELoss = nn.MSELoss()

        printc("\nLoading camembert and its tokenizer...", "INFO")
        self.camembert = CamembertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels)
        self.camembert.to(self.device)

        self.ratio_lr_embeddings = config.ratio_lr_embeddings if config.ratio_lr_embeddings else 1
        decomposed_params = [{'params': self.camembert.roberta.embeddings.parameters(), 'lr': self.learning_rate*self.ratio_lr_embeddings},
                        {'params': self.camembert.roberta.encoder.parameters()},
                        {'params': self.camembert.classifier.parameters()}]
        self.optimizer = Adam(decomposed_params, lr = self.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer)

        if config.freeze:
            self.freeze()

        self.tokenizer = CamembertTokenizerFast.from_pretrained(self.model_name)

        if config.resume:
            self.resume(config)
        printc("Successfully loaded\n", "SUCCESS")

        self.drop_rate = config.drop_rate
        if self.drop_rate:
            set_dropout(self.camembert, drop_rate=self.drop_rate)
            print(f"Dropout rate set to {self.drop_rate}")

        if self.voc_path:
            self.add_tokens_from_path(self.voc_path)

        self.eval()

    def resume(self, config):
        printc(f"Resuming with model at {config.resume}...", "INFO")
        path_checkpoint = os.path.join(os.path.dirname(config.path_result), config.resume, 'checkpoint.pth')
        assert os.path.isfile(path_checkpoint), 'Error: no checkpoint found!'
        checkpoint = torch.load(path_checkpoint, map_location=self.device)
        self.tokenizer = checkpoint['tokenizer']
        self.camembert.resize_token_embeddings(len(self.tokenizer))

        try:
            self.load_state_dict(checkpoint['model'])
        except:
            printc('Resuming from a model trained on a different mode. The last classification layer has to be trained again.', 'WARNING')
            for parameter in checkpoint['model'].keys():
                if parameter.split('.')[2] == 'out_proj':
                    checkpoint['model'][parameter] = self.state_dict()[parameter]
            self.load_state_dict(checkpoint['model'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

    def freeze(self):
        """Freezes the encoder layer. Only the classification head on top will learn"""
        self.frozen = True
        for param in self.camembert.roberta.parameters():
            param.requires_grad = False

    def forward(self, *input, **kwargs):
        """Camembert forward for classification or regression"""
        if self.mode == 'classif':
            return self.camembert(*input, **kwargs)
        elif self.mode == 'regression':
            return torch.sigmoid(self.camembert(*input, **kwargs).logits)
        else:
            logits = self.camembert(*input, **kwargs).logits
            mus = torch.sigmoid(logits[:,0])
            log_vars = logits[:,1]
            return mus, log_vars

    def get_loss(self, outputs, labels=None):
        """Returns the loss given outputs and labels"""
        if self.mode == 'classif':
            return outputs.loss
        elif self.mode == 'regression':
            return self.MSELoss(outputs.reshape(-1), labels)
        else:
            mu, log_var = outputs
            return (log_var + (labels - mu)**2/torch.exp(log_var)).mean()

    def step(self, texts, labels=None, output_hidden_states=False):
        """
        Encode and forward the given texts. Compute the loss, and its backward.

        Args:
            texts (str list)
            labels (tensor): tensor of 0-1 (classification) or float (regression)

        Outputs:
            loss
            camembert outputs
        """
        encoding_start_time = time()
        encoding = self.tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True)
        self.encoding_time += time()-encoding_start_time

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        if not labels is None:
            if not self.mode == 'classif':
                labels = labels.type(torch.FloatTensor)
            labels = labels.to(self.device)

        compute_start_time = time()
        outputs = self(input_ids, attention_mask=attention_mask, labels=(labels if self.mode == 'classif' else None), output_hidden_states=output_hidden_states)
        self.compute_time += time()-compute_start_time

        if output_hidden_states: return outputs.hidden_states[-1][:,0,:]

        if labels is None: return outputs.logits if self.mode == 'classif' else outputs

        loss = self.get_loss(outputs, labels=labels)

        if self.camembert.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        

        return loss, outputs.logits if self.mode == 'classif' else outputs

    def add_tokens_from_path(self, voc_path):
        """
        Read a file of vocabulary and add the given tokens into the model

        Args:
            voc_path: path to a json file whose keys are words
        """
        with open(voc_path) as json_file:
            voc_list = json.load(json_file)
            new_tokens = self.tokenizer.add_tokens([ token for (token, _) in voc_list ])
            print(f"Added {new_tokens} tokens to the tokenizer")

        self.camembert.resize_token_embeddings(len(self.tokenizer))

    def train(self):
        """Training mode"""
        self.camembert.train()

    def eval(self):
        """Eval mode (no random)"""
        self.camembert.eval()

