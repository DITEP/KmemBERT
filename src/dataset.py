'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
import os
import sys
from collections import defaultdict
from tqdm import tqdm

from .utils import get_label, time_survival_to_label, printc
from .preprocesser import EHRPreprocesser
from .models import HealthBERT

class EHRDataset(Dataset):
    """PyTorch Dataset class for EHRs"""

    def __init__(self, path_dataset, config, train=True, df=None):
        super(EHRDataset, self).__init__()
        self.path_dataset = path_dataset
        self.config = config
        self.train = train
        self.config_path = os.path.join(self.path_dataset, "config.json")
        self.preprocesser = EHRPreprocesser()

        if df is not None:
            self.df = df
        else:
            self.csv_path = os.path.join(self.path_dataset, "train.csv" if train else "test.csv")
            self.df = pd.read_csv(self.csv_path)

        self.survival_times = np.array(list(self.df[["Date deces", "Date cr"]].apply(lambda x: get_label(*x), axis=1)))

        if os.path.isfile(self.config_path):
            with open(self.config_path) as json_file:
                self.mean_time_survival = json.load(json_file)["mean_time_survival"]
        elif self.train:
            pass
        else:
            sys.exit("config.json is needed for testing. Exiting..")
        config.mean_time_survival = self.mean_time_survival

        self.labels = time_survival_to_label(self.survival_times, self.mean_time_survival)
        self.texts = list(self.df["Texte"].astype(str).apply(self.preprocesser))
        
    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

    @classmethod
    def get_train_validation(cls, path_dataset, config, get_only_val=False, **kwargs):
        """
        Returns train and validation set based on a predefined split
        """
        df = pd.read_csv(os.path.join(path_dataset, "train.csv"))
        validation_split = pd.read_csv(os.path.join(path_dataset, "validation_split.csv"), dtype=bool)
        
        validation = df[validation_split["validation"]]
        train = df.drop(validation.index)

        if get_only_val:
            return cls(path_dataset, config, df=validation, **kwargs)
        return cls(path_dataset, config, df=train, **kwargs), cls(path_dataset, config, df=validation, **kwargs)

class PredictionsDataset(EHRDataset):
    """
    Dataset dealing with multiple EHRs instead of only one.
    The models defined inside models.multi_ehr needs this dataset.
    """
    health_bert = None

    def __init__(self, *args, device=None, output_hidden_states=False, **kwargs):
        super(PredictionsDataset, self).__init__(*args, **kwargs)
        self.device = device
        self.output_hidden_states = output_hidden_states
        self.load_health_bert()

        self.patients = list(set(self.df.Noigr.values))
        self.noigr_to_indices = defaultdict(list)
        self.length = 0
        for i, noigr in enumerate(self.df.Noigr.values):
            self.noigr_to_indices[noigr].append(i)
            self.length += 1
            if self.config.nrows and self.length == self.config.nrows:
                break

        for noigr, indices in self.noigr_to_indices.items():
            self.noigr_to_indices[noigr] = sorted(indices, key=lambda i: -self.survival_times[i])

        self.index_to_ehrs = [(noigr, k) for noigr, indices in self.noigr_to_indices.items() for k in range(1, len(indices)+1)]

        self.compute_prediction()

    def load_health_bert(self):
        if PredictionsDataset.health_bert is None:
            if self.output_hidden_states:
                self.config.mode = "classif"

            PredictionsDataset.health_bert = HealthBERT(self.device, self.config)
            for param in self.health_bert.parameters():
                param.requires_grad = False

    def compute_prediction(self):
        printc('\nComputing Health Bert predictions...', 'INFO')
        self.noigr_to_outputs = defaultdict(list)
        for noigr, indices in tqdm(self.noigr_to_indices.items()):
            for index in indices:
                output = self.health_bert.step([self.texts[index]], output_hidden_states=self.output_hidden_states)
                self.noigr_to_outputs[noigr].append(output)
        printc(f'Successfully computed {self.length} Health Bert outputs\n', 'SUCCESS')

    def __getitem__(self, index):
        noigr, k = self.index_to_ehrs[index]

        indices = self.noigr_to_indices[noigr][:k][-self.config.max_ehrs:]
        dt = self.survival_times[indices] - min(self.survival_times[indices])
        label = min(self.labels[indices])

        outputs = self.noigr_to_outputs[noigr][:k][-self.config.max_ehrs:]

        if self.output_hidden_states:
            return (torch.cat(outputs), dt, label)

        elif self.config.mode == 'density':
            mus = torch.cat([output[0] for output in outputs]).view(-1)
            log_vars = torch.cat([output[1] for output in outputs]).view(-1)
            return (mus, log_vars, dt, label)

        else:
            mus = torch.cat(outputs).view(-1)
            return (mus, dt, label)
        
    def __len__(self):
        return self.length