# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import pandas as pd
import os

from utils import get_label

class TweetDataset(Dataset):
    """PyTorch Dataset class for tweets"""

    def __init__(self, csv_name):
        super(TweetDataset, self).__init__()
        self.csv_name = csv_name
        self.df = pd.read_csv(self.csv_name)
        self.labels = list(self.df.label)
        self.texts = list(self.df.text)
        
    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

class EHRDataset(Dataset):
    """PyTorch Dataset class for tweets"""

    def __init__(self, csv_name):
        super(EHRDataset, self).__init__()
        self.csv_name = csv_name
        self.df = pd.read_csv(self.csv_name, sep='£', engine='python')
        # TODO: labels_to_uniform
        self.labels = list(self.df[["Date deces", "Date cr"]].apply(lambda x: get_label(*x), axis=1))
        self.texts = list(self.df["Texte"])
        
    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def __len__(self):
        return len(self.labels)