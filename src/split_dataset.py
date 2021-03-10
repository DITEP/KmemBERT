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
from utils import save_json, get_label, get_date


train_size = 0.7
seed = 0

data_path = "/data/isilon/centraleNLP"
file_path = os.path.join(data_path, "concatenate.txt")
train_path = os.path.join(data_path, "train.csv")
test_path = os.path.join(data_path, "test.csv")

if os.path.isfile(train_path) or os.path.isfile(test_path):
    raise BaseException("File exists: can't overwrite existing train and test datasets.")


print("Reading csv...")
def f_line(line):
    row = line.split('£')
    if row[-1][-1] == "\n":
        row[-1] = row[-1][:-1] # get rid of \n
    elif row[-1][-2:] == "\n\r":
        row[-1] = row[-1][:-2] # get rid of \n\r
    return row 

rows = list(filter(lambda x: len(x)==9, [f_line(line) for line in open(file_path)]))
df = pd.DataFrame(rows[1:], columns=rows[0])
print(f"df shape: {df.shape}")

print("\nCounting EHR categories...\n")
counter = df.groupby("Nature doct").count()["Noigr"]
print(counter)

print("\nFiltering EHR...")
# 2904066
df = df[df["Nature doct"] == "C.R. consultation"]
print(f"{df.shape[0]} rows left")
# 1347612
df["Texte"].replace("", np.nan, inplace=True)
df["Date deces"].replace("", np.nan, inplace=True)
df["Date cr"].replace("", np.nan, inplace=True)
df["Noigr"].replace("", np.nan, inplace=True)
df.dropna(subset=["Date deces", "Date cr", "Texte", "Noigr"], inplace=True)
print(f"{df.shape[0]} rows left")
# 1347607
df = df[df["Date cr"]<df["Date deces"]]
print(f"{df.shape[0]} rows left")
# 1343113

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

# Split
noigrs = pd.unique(df["Noigr"])
train_noigrs, test_noigrs = train_test_split(noigrs, train_size=train_size, random_state=seed)

train=df[df["Noigr"].isin(train_noigrs)]
test=df.drop(train.index)
mean_time_survival = np.mean(list(train[["Date deces", "Date cr"]].apply(lambda x: get_label(*x), axis=1)))

def save_df(df, path):
    df.to_csv(path, index=False)
    # CRAPPY
    # I don't know why when loading the dataset "new" missing values appear
    df = pd.read_csv(path)
    n = df.shape[0]
    df.dropna(subset=["Date deces", "Date cr", "Texte", "Noigr"], inplace=True)
    print(f"Removed {n-df.shape[0]} rows")
    df.to_csv(path, index=False)

print("\nSaving train...")
save_df(train, train_path)
print("\nSaving test...")
save_df(test, test_path)
save_json(data_path, "config", {"mean_time_survival": mean_time_survival})
n_train, n_test = len(train), len(test)
print("\nTrain samples: {}\nTest samples: {}\nTrain ratio: {}".format(n_train, n_test, n_train/(n_train + n_test)))