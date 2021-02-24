"""
RUN ONLY ONCE
This code splits the concatenated dataset into a train and test one.
"""
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
import sys
sys.path.append("../src/")
from utils import get_label, save_json

train_size = 0.7
seed = 0

data_path = "/data/isilon/centraleNLP"
file_path = os.path.join(data_path, "concatenate.txt")
train_path = os.path.join(data_path, "train.csv")
test_path = os.path.join(data_path, "test.csv")


if os.path.isfile(train_path) or os.path.isfile(test_path):
    raise BaseException("File exists: can't overwrite existing train and test datasets.")


print("Reading csv...")
df = pd.read_csv(file_path, sep='\xc2\xa3', engine='python')

print("\nCounting EHR categories...\n")
counter = df.groupby("Nature doct").count()["Noigr"]
print(counter)

print("\nFiltering EHR...")
df = df[df["Nature doct"] == "C.R. consultation"]
df.dropna(subset=["Date deces", "Date cr", "Texte", "Noigr"], inplace=True)
df = df[df["Date cr"]<df["Date deces"]]

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

# Split
noigrs = pd.unique(df["Noigr"])
train_noigrs, test_noigrs = train_test_split(noigrs, train_size=train_size, random_state=seed)

train=df[df["Noigr"].isin(train_noigrs)]
test=df.drop(train.index)
mean_time_survival = np.mean(list(train[["Date deces", "Date cr"]].apply(lambda x: get_label(*x), axis=1)))

train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)
save_json(data_path, "config", {"mean_time_survival": mean_time_survival})
n_train, n_test = len(train), len(test)
print("\nTrain samples: {}\nTest samples: {}\nTrain ratio: {}".format(n_train, n_test, n_train/(n_train + n_test)))