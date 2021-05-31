import pandas as pd
from collections import defaultdict
import argparse
import os
import numpy as np

from kmembert.utils import get_label

survival_times_dict = defaultdict(int)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
    help="data folder name")
args = parser.parse_args()

df = pd.read_csv(os.path.join('data', args.data_folder, 'train.csv'))

survival_times = np.array(list(df[["Date deces", "Date cr"]].apply(lambda x: get_label(*x), axis=1)))

for id, time in zip(df.Noigr, survival_times):
    survival_times_dict[id] = max(survival_times_dict[id], time)

print("Mean follow up:", np.mean(list(survival_times_dict.values())), "days")