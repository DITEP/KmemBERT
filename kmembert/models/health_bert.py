'''
    Author: CentraleSupelec & Théo Di Piazza
    Year: 2022
    Python Version: >= 3.7
'''

import json
import os
from transformers import CamembertModel, CamembertTokenizerFast
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

from ..utils import printc, my_custom_loss
from .interface import ModelInterface

class CamembertRegressor(ModelInterface):
    """
    Model that instanciates a camembert model, a tokenizer and an optimizer.
    It supports methods to train it.
    It is used for regression task
    """
    def __init__(self, device, config):
        super(CamembertRegressor, self).__init__(device, config)
        
        # Get parameters from config
        self.learning_rate = config.learning_rate

        # Input, Output dimension of the output layer
        self.D_in, self.D_out = 768, config.num_label
        self.drop_rate = config.drop_rate

        printc("\nLoading camembert and its tokenizer...", "INFO")
        # Camembert Layers
        self.camembert = CamembertModel.from_pretrained('camembert-base')
        # Camembert Tokenizer
        self.tokenizer = CamembertTokenizerFast.from_pretrained('camembert-base')
        # Output Layer : Regression Task
        self.fc1 = nn.Sequential( # Premiere classification
            nn.Dropout(self.drop_rate),
            nn.Linear(self.D_in, self.D_out))

        # Optimizer
        decomposed_params = self.parameters()
        self.ratio_lr_embeddings = config.ratio_lr_embeddings if config.ratio_lr_embeddings else 1

        # Permits to have specific hyper-parameters per layer 
        decomposed_params = [{'params': self.camembert.embeddings.parameters(), 'lr': self.learning_rate*self.ratio_lr_embeddings},
                        {'params': self.camembert.encoder.parameters()},
                        {'params': self.fc1.parameters()}]
        self.optimizer = Adam(decomposed_params, lr = self.learning_rate, weight_decay=config.weight_decay)
        
        # Loss function of the model: MSE for Regression Task
        #self.Loss = nn.MSELoss()
        #self.Loss = Custom_Loss()
        self.Loss = nn.BCEWithLogitsLoss()

        printc("Successfully loaded\n", "SUCCESS")

    # Forward function: To get output from input
    def forward(self, *input, **kwargs):
        """Camembert forward for regression
        Inputs:
            input: input_ids, attention_mask
        Output:
            Vector with continuous features
        """
        # First layers: Camembert
        outputs = self.camembert(*input, **kwargs)
        class_label_output = outputs[1]

        # Final layers: For Multi Regression Task
        outputs = self.fc1(class_label_output)
        
        return outputs
    
    # Step function: To train the model
    def step(self, texts, labels=None, output_hidden_states=False):
        """
        Encode and forward the given texts. Compute the loss, and its backward.
        Args:
            texts (str list)
            labels (tensor): Approximated probability
        Outputs:
            loss
            camembert outputs
        """
        # Tokenize the input: the text
        encoding = self.tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Get prediction: outputs
        outputs = self(input_ids, attention_mask=attention_mask)
        print(labels)
        print("shape de labels:", labels.shape)
        # Get Loss from outputs and labels
        #loss = self.Loss(outputs, labels)
        loss = my_custom_loss(outputs, labels)
        #loss.requires_grad = True

        # Update optimizer
        if self.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss, outputs

