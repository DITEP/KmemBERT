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
from .models import CamembertRegressor

class EHRDataset(Dataset):
    """PyTorch Dataset class for EHRs"""

    def __init__(self, path_dataset, config, train=True, df=None, return_id=False):
        super(EHRDataset, self).__init__()
        self.path_dataset = path_dataset
        self.config = config
        self.train = train
        self.return_id = return_id
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

        # Initialize inputs and outputs of the dataset
        # Inputs: the text
        self.texts = list(self.df["Texte"].astype(str).apply(self.preprocesser))

        # Outputs: Labels for each Time Interval
        self.num_label = config.num_label
        zip_columns = zip(df["f1"])
        # Add each column (IT: Time Interval) to 'the labels' feature
        for i in range(1, self.num_label):
            zip_columns = zip(*zip(*zip_columns), df["f"+str(i+1)])
        self.labels = torch.tensor(list(map(list, zip_columns)), dtype=float)
        
    def __getitem__(self, index):
        """
        if self.return_id :
            return self.texts[index], self.noigr[index], (self.labels1[index], self.labels2[index])
        """
        return self.texts[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

    @classmethod
    def get_train_validation(cls, path_dataset, config, get_only_val=False, **kwargs):
        """
        Returns train and validation set based on a predefined split
        """
        df = pd.read_csv(os.path.join(path_dataset, "train.csv"))

        df["f1"]=[1]*18; df["f2"]=[0]*9+[1]*9; df["f3"]=[0]*9+[-1]*9

        validation_split = pd.read_csv(os.path.join(path_dataset, "validation_split.csv"), dtype=bool)
        
        validation = df[validation_split["validation"]]
        train = df.drop(validation.index)

        if get_only_val:
            return cls(path_dataset, config, df=validation, **kwargs)
        return cls(path_dataset, config, df=train, **kwargs), cls(path_dataset, config, df=validation, **kwargs)