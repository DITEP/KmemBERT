"""
RUN ONLY ONCE
This code splits the concatenated dataset into a train and test one.
"""
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import date
import json

# CRAPPY: copy from utils but works for python 2.7
# TODO: make it work for python 3.6, probably need to change concatenate_files.py as well, or do a utils.py without f strings
def save_json(path_result, name, x):
    with open(os.path.join(path_result, "{}.json".format(name)), 'w') as f:
        json.dump(x, f, indent=4)

def get_date(str_date):
    """
    Being given the string 20160211 returns date(2016,2,11)
    """
    year = int(str_date[:4])
    month = int(str_date[4:6])
    day = int(str_date[6:8])
    return date(year, month, day)

def get_label(str_date_deces, str_date_cr):
    """
    Being given 2 strings like 20160201 and 20170318 returns the corresponding time difference in number of days.
    Date format: yyyymmdd
    """
    
    date_deces = get_date(str(str_date_deces))
    date_cr = get_date(str(str_date_cr))

    delta = date_deces - date_cr
    return delta.days

train_size = 0.7
seed = 0

data_path = "/data/isilon/centraleNLP"
file_path = os.path.join(data_path, "concatenate.txt")
train_path = os.path.join(data_path, "train.csv")
test_path = os.path.join(data_path, "test.csv")
custom_separator = 'secrettoken749386453728394027'


if os.path.isfile(train_path) or os.path.isfile(test_path):
    raise BaseException("File exists: can't overwrite existing train and test datasets.")


print("Reading csv...")
def f_line(line):
    row = line.split('£')
    row[-1] = row[-1][:-1] # get rid of \n
    return row 

rows = list(filter(lambda x: len(x)==9, [f_line(line) for line in open(file_path)]))
df = pd.DataFrame(rows[1:], columns=rows[0])

print("\nCounting EHR categories...\n")
counter = df.groupby("Nature doct").count()["Noigr"]
print(counter)

print("\nFiltering EHR...")
# 2904066
df = df[df["Nature doct"] == "C.R. consultation"]
# 1347612
df.dropna(subset=["Date deces", "Date cr", "Texte", "Noigr"], inplace=True)
# 1347572
df = df[df["Date cr"]<df["Date deces"]]

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

# Split
noigrs = pd.unique(df["Noigr"])
train_noigrs, test_noigrs = train_test_split(noigrs, train_size=train_size, random_state=seed)

train=df[df["Noigr"].isin(train_noigrs)]
test=df.drop(train.index)
mean_time_survival = np.mean(list(train[["Date deces", "Date cr"]].apply(lambda x: get_label(*x), axis=1)))

np.savetxt(train_path, train, delimiter=custom_separator, fmt='%s')
np.savetxt(test_path, test, delimiter=custom_separator, fmt='%s')
# train.to_csv(train_path, index=False)
# test.to_csv(test_path, index=False)
save_json(data_path, "config", {"mean_time_survival": mean_time_survival})
n_train, n_test = len(train), len(test)
print("\nTrain samples: {}\nTest samples: {}\nTrain ratio: {}".format(n_train, n_test, n_train/(n_train + n_test)))