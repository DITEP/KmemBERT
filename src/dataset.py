from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
import os
import sys

from utils import get_label, time_survival_to_label
from preprocesser import EHRPreprocesser

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
    """PyTorch Dataset class for EHRs"""

    def __init__(self, path_dataset, config, train=True):
        super(EHRDataset, self).__init__()
        self.path_dataset = path_dataset
        self.nrows = config.nrows
        self.train = train
        self.csv_path = os.path.join(self.path_dataset, "train.csv" if train else "test.csv")
        self.config_path = os.path.join(self.path_dataset, "config.json")
        self.preprocesser = EHRPreprocesser()

        self.df = pd.read_csv(self.csv_path)

        self.labels = np.array(list(self.df[["Date deces", "Date cr"]].apply(lambda x: get_label(*x), axis=1)))

        if os.path.isfile(self.config_path):
            with open(self.config_path) as json_file:
                self.mean_time_survival = json.load(json_file)["mean_time_survival"]
        elif self.train:
            pass
        else:
            sys.exit("config.json is needed for testing. Exiting..")
        config.mean_time_survival = self.mean_time_survival

        self.labels = time_survival_to_label(self.labels, self.mean_time_survival)
        self.texts = list(self.df["Texte"].apply(self.preprocesser))
        
    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def __len__(self):
        return len(self.labels)