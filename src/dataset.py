# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import pandas as pd
import os

from utils import get_label

class TweetDataset(Dataset):
    """PyTorch Dataset class for tweets"""

    def __init__(self, csv_name, config):
        super(TweetDataset, self).__init__()
        self.csv_name = csv_name
        self.nrows = config.nrows
        self.df = pd.read_csv(self.csv_name, nrows=self.nrows)
        self.labels = list(self.df.label)
        self.texts = list(self.df.text)

        print("labels--", len(self.labels))
        
    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

class EHRDataset(Dataset):
    """PyTorch Dataset class for tweets"""

    def __init__(self, csv_name, config):
        super(EHRDataset, self).__init__()
        self.csv_name = csv_name
        self.nrows = config.nrows
        self.df = pd.read_csv(self.csv_name, sep='£', engine='python', nrows=self.nrows)
        # TODO: labels_to_uniform
        self.labels = list(self.df[["Date deces", "Date cr"]].apply(lambda x: get_label(*x), axis=1))
        self.texts = list(self.df["Texte"])
        
    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def __len__(self):
        return len(self.labels)