'''
Author : Théo Di Piazza (CLB)
Year : 2022

    Script to preprocess CLB data from n extracted patients dataset.
    Script to extract these n patients : 'testT2_data_prepare.py'
    Dataframe read: doc_toTest_YEAR-MONTH-DAY_nNBPATIENT.csv
    Dataframe output: doc_toTest_YEAR-MONTH-DAY_nNBPATIENT_preproc.csv

    Please change 'date_df' (name of the df to read) and nb_patient (number of extracted
    patients) according to your need.

    Command line to execute: python testT2_data_preprocessing.py
'''

import pandas as pd
import os
from utils import printc
os.system('color')

printc("    Preprocess data to test T2 format...", "WARNING")

###############################################
# Read the dataframe
date_df = "2022-05-04"; nb_patient = 100
df_to_read = "doc_toTest\\doc_toTest_" + date_df + "_n" + str(nb_patient)
df = pd.read_csv(df_to_read + ".csv")

###############################################
# Pre-processing complet sur donnees CLB

# Remove the first sentance with no information
df.Texte = df.Texte.apply(lambda x: x[x.find('.')+2:])

# Remove '\xa0'
df.Texte = df.Texte.apply(lambda x: x.replace("\xa0",""))

# Filtre sur les documents avec plus de 250 caracteres
df['Longueur'] = df.Texte.apply(lambda x: len(x))
df = df[df.Longueur >= 250]

# Filtre sur les documents avec trop de balises HTML
df['HTML_count'] = df.Texte.apply(lambda x: x.count('</'))
df = df[df.HTML_count <= 2]

# Select usefull columns for the test model
df = df[['Noigr', 'Date cr', 'Date deces', 'Texte']]

###############################################
# Save the file as csv
name_file =  df_to_read + "preprocessed.csv"
df.to_csv(name_file, index=False)

printc("    Preprocessing of data succeeded.", "SUCCESS")
###############################################
# END







