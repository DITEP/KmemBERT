# -*- coding: utf-8 -*-
"""
RUN ONLY ONCE
This code splits the concatenated dataset into a train and test one.
"""
import pandas as pd 
import os

train_size = 0.7
seed = 0

data_path = "/data/isilon/centraleNLP"
file_path = os.path.join(data_path, "concatenate.txt")
train_path = os.path.join(data_path, "train_dataset.csv")
test_path = os.path.join(data_path, "test_dataset.csv")


if os.path.isfile(train_path) or os.path.isfile(test_path):
    raise BaseException("File exists: can't overwrite existing train and test datasets.")


df = pd.read_csv(file_path, sep='£', error_bad_lines=False, engine='python')

train=df.sample(frac=train_size, random_state=seed) 
test=df.drop(train.index)

train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)