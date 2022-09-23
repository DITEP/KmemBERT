'''
Author : Théo Di Piazza (CLB)
Year : 2022

    Script to extract n patients to test T2
    Dataframe read: doc_df_CRC.29042022.csv (to change please testT2_data_cleaning.ipynb)
    Dataframe output: doc_toTest_YEAR-MONTH-DAY_nNBPATIENT.csv

    Please, change 'n' (number of patients extracted according to your need).
    Please, change 'date_wanted_file' to get a specific dataframe
    from testT2_data_read_clean.py.

    Command line to execute: python testT2_data_prepare.py
'''

import pandas as pd
import random
import os
from datetime import datetime
from utils import printc
os.system('color')

random.seed(2022)

printc("    Extract n patients to test T2 format...", "WARNING")
###############################################
# Read dataframe
date_wanted_file = "2022-05-04"
df_to_read = "doc_toTest\\doc_df_CRC_" + date_wanted_file + ".csv"
df = pd.read_csv(df_to_read)

###############################################
# Filter on dead patients
df = df[df.FLAG_DECES == 1]

# Extract n patients from the dataset
n = 100
# Get IPPR of extracted patients
IPPR_sample = random.sample(df.IPPR.tolist(), n)
# Get documents of these patients
df_sample = df[df.IPPR.isin(IPPR_sample)]
# Rename with good col names
df_sample = df_sample.copy()
df_sample.rename(columns={'IPPR': 'Noigr', 'DCREA': 'Date cr', 
                    'DECES': 'Date deces', 'text': 'Texte'}, inplace=True)
# Remove hyphens from date columns
df_sample['Date cr'] = df_sample['Date cr'].astype(str).apply(lambda x: x.replace('-', '')).astype(int)
df_sample['Date deces'] = df_sample['Date deces'].astype(str).apply(lambda x: x.replace('-', '')).astype(int)
# Get usefull columns
df_res = df_sample[['Noigr', 'Date cr', 'Date deces', 'Texte']]

###############################################
# Save the file as csv
name_file = "doc_toTest\\doc_toTest_" + datetime.today().strftime('%Y-%m-%d') + "_n" + str(n) + ".csv"
df_res.to_csv(name_file, index=False)

printc("    Extraction of n patients succeeded. Please preprocess before testing.", "SUCCESS")
###############################################
# END