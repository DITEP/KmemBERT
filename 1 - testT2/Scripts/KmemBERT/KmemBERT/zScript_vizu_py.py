'''
    Python script to interprete prediction of a patient

    python3 zScript_vizu_py.py
'''

# Import libraries
import numpy as np
import pandas as pd
import argparse
from torch.serialization import save
from transformers_interpret import SequenceClassificationExplainer
import matplotlib.pyplot as plt

from kmembert.models import HealthBERT
from kmembert.utils import create_session

# Import argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_folder", type=str, default="data/ehr/test.csv", 
    help="data path to access to the testing file")
parser.add_argument("-p", "--path_dataset", type=str, default="data/ehr/test.csv", 
    help="data path to access to the testing file")
parser.add_argument("-r", "--resume", type=str, default="kmembert-base", 
    help="result folder in with the saved checkpoint will be reused")
parser.add_argument("-nr", "--nrows", type=int, default=10, 
    help="maximum number of samples for testing")
parser.add_argument("-f", "--folder_to_save", type=str, default="graphs", 
    help="folder to save the figures")
parser.add_argument("-ng", "--noigr", type=int, default=2, 
    help="The Noigr of a patient")
args = parser.parse_args("")

# Load Model
_, _, device, config = create_session(args)
model = HealthBERT(device, config)

# Read df used to test
# Creation du jeu de donnees pour retrouver les infos interessantes
#test = pd.read_csv("kmembert\\Visualization\\test_VM.csv")
test = pd.read_csv("kmembert/Visualization/test_VM.csv")

# Read df with prediction
#res_pred = pd.read_csv("kmembert\\Visualization\\results_pred_VM.csv")
res_pred = pd.read_csv("kmembert/Visualization/results_pred_VM.csv")

# Merge 2 dataframes
resul_df = pd.merge(test, res_pred, left_on="indice", right_on="ind")[['Noigr', 'Date cr', 'Date deces', 'Texte', 'indice', 'pred', 'lab']]

# Add columns to have absolute error between pred and lab
resul_df['ecart'] = abs(resul_df['pred']-resul_df['lab'])

resul_df['nb_words'] = resul_df['Texte'].apply(lambda x: len(x.split()))

resul_df['nb_docs'] = resul_df.groupby(["Noigr"])["Noigr"].transform("count")

resul_df = resul_df.sort_values("ecart")

# Select Noigr and Indice's CR to select text to study
id_noigr = 9110391
id_indice = 21338

# Create a list texts_to_classify which contains CR texts to classify
texts_to_classify = resul_df[(resul_df.Noigr==id_noigr) & (resul_df.indice<=id_indice) & (resul_df.indice>id_indice-4)].sort_values("indice").Texte.values
import math
n_max = 1500
for i in range(len(texts_to_classify)):
    if(len(texts_to_classify[i])>n_max):
        texts_to_classify[i] = texts_to_classify[i][:n_max]
#texts_to_classify = texts_to_classify[0]

################################################################
# Read medical vocabulary
import json
f = open("medical_voc/large.json", encoding='utf-8')
dictio = json.load(f)
med_voc = []
for i in range(len(dictio)):
    med_voc.append(dictio[i][0])
################################################################
# Filter on too long docs, otherwise there will not be read
n_max = 1500
################################################################
# For each 'clean' text, compute word attributions
word_attributes = []
for ehr in texts_to_classify:
    cls_explainer = SequenceClassificationExplainer(
        model.camembert,
        model.tokenizer)
    word_attributions = cls_explainer(ehr)
    word_attributes+= [dict(word_attributions)]
################################################################
# For each word, sum attributions for each word (without duplicates keys)
result = {}
n_documents = len(word_attributes)
for d in word_attributes:
    for k in d.keys():
        result[k] = result.get(k, 0) + d[k]/n_documents
################################################################
# Create dict to stock word+attributions for words in medical voc
result_medicam = {}
# For each word of the text to classify
for k, v in result.items():
    # If special character present, replace it with lower case
    new_k = k
    if("▁" in k):
        new_k = k.replace("▁", "").lower()
    elif("_" in k):
        new_k = k.replace("_", "").lower()
    # Add word to dict with transformed string, if in medical voc
    if new_k in med_voc:
        result_medicam[new_k] = v
################################################################
# Sort words by attributions value and show 10 most importants
show_n = 10
final_dict = dict(sorted(result_medicam.items(), key=lambda item: abs(item[1]), reverse=True)[:show_n])
################################################################
# Plot the results for the word attributes (green for positive contribution and red for negative contribution)
x = list(final_dict.keys())
y = list(final_dict.values())
colors = ['deeppink']*show_n 
colors = ['deeppink']*len(y) 
for i in range(len(y)):
    if y[i]>0:
        colors[i] = 'dodgerblue'
plt.figure()
plt.barh(list(reversed(x)), list(reversed(y)), color=list(reversed(colors)))
plt.title(f'Word Attributions for all EHRs of the patient {id_noigr}')
plt.savefig(f'graphs/interpretation/test_{id_noigr}.png', bbox_inches='tight')    
plt.close()
