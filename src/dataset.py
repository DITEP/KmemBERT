'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import os
import sys
from collections import defaultdict

from .utils import get_label, time_survival_to_label, printc
from .preprocesser import EHRPreprocesser
from .models.health_bert import HealthBERT

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
    def get_train_validation(cls, path_dataset, config):
        """
        Returns train and validation set based on a predefined split
        """
        df = pd.read_csv(os.path.join(path_dataset, "train.csv"))
        validation_split = pd.read_csv(os.path.join(path_dataset, "validation_split.csv"), dtype=bool)
        
        validation = df[validation_split["validation"]]
        train = df.drop(validation.index)

        return cls(path_dataset, config, df=train), cls(path_dataset, config, df=validation)

class EHRHistoryDataset(EHRDataset):
    """PyTorch Dataset class for a full history of EHRs"""

    def __init__(self, *args, **kwargs):
        super(EHRHistoryDataset, self).__init__(*args, **kwargs)

        self.patients = list(set(self.df.Noigr.values))
        self.patient_to_indices = defaultdict(list)
        for i, noigr in enumerate(self.df.Noigr.values):
            self.patient_to_indices[noigr].append(i)

        for noigr, indices in self.patient_to_indices.items():
            self.patient_to_indices[noigr] = sorted(indices, key=lambda i: -self.survival_times[i])

        self.index_to_ehrs = [(noigr, k) for noigr, indices in self.patient_to_indices.items() for k in range(1, len(indices)+1)]
    
    def __getitem__(self, index):
        noigr, k = self.index_to_ehrs[index]

        indices = self.patient_to_indices[noigr][:k][-self.config.max_ehrs:]
        last_survival_time = min(self.survival_times[indices])

        return ([self.texts[text_index] for text_index in indices], 
                self.survival_times[indices] - last_survival_time, 
                min(self.labels[indices]))

    def __len__(self):
        return len(self.index_to_ehrs)

def collate_fn(batch):
    sequences, times, labels = zip(*batch)
    return sequences, torch.tensor(times).type(torch.float32), torch.tensor(labels).type(torch.float32)

class PredictionsDataset(Dataset):
    health_bert = None

    def __init__(self, device, config, history_dataset):
        self.device = device
        self.config = config
        self.history_dataset = history_dataset
        self.loader = DataLoader(self.history_dataset, batch_size=1, collate_fn=collate_fn)

        self.load_health_bert()
        self.compute_prediction()

    def load_health_bert(self):
        if PredictionsDataset.health_bert is None:
            with open(os.path.join('results', self.config.resume, 'args.json')) as json_file:
                self.config.mode = json.load(json_file)["mode"]
                assert self.config.mode != "classify", "Health Bert mode classify not supported for RNNs"
                printc(f"\nUsing mode {self.config.mode} (Health BERT checkpoint {self.config.resume})", "INFO")

            PredictionsDataset.health_bert = HealthBERT(self.device, self.config)
            for param in self.health_bert.parameters():
                param.requires_grad = False

    def compute_prediction(self):
        printc('\nComputing Health Bert predictions...', 'INFO')
        self.predictions = []
        for (texts, dt, label) in self.loader:
            dt = dt.to(self.device)[0]

            if self.config.mode == 'density':
                mus, log_vars = self.health_bert.step(texts[0])
                self.predictions.append((mus, log_vars, dt, label[0]))
            else:
                mus = self.health_bert.step(texts[0])
                self.predictions.append((mus, dt, label[0]))
        printc(f'Successfully computed {len(self.predictions)}/{len(self.history_dataset)} predictions\n', 'SUCCESS')

    def __getitem__(self, index):
        return self.predictions[index]
        
    def __len__(self):
        return len(self.predictions)