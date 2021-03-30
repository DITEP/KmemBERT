'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import re 
import datetime
import numpy as np
from numpy.core.numeric import NaN
from transformers import CamembertTokenizerFast

from ..preprocesser import EHRPreprocesser

def count_sentence(x, tokenizer, preprocesser):
    count_token = 0
    x["Texte"] = preprocesser(x["Texte"])
    nb_tokens = len(tokenizer(x["Texte"], max_length =2000)["input_ids"])
    splitted_by_sentence = re.split(r'[.!?]+', str(x["Texte"]))
    total_sentence = len(splitted_by_sentence) - 1
    for s in splitted_by_sentence :
        if len(s) == 0 :
            continue 
        count_token += (len(tokenizer(s)["input_ids"]) - 1)
    mean_tokens = count_token / max(1, total_sentence)
    return pd.Series([total_sentence, mean_tokens, nb_tokens], index = ["total_sentence", "mean_tokens", "nb_tokens"])

def compute_survival_time(x):
    count_dates_inversed = 0
    date_CR = str(int(x["Date cr"]))
    date_death = str(int(x["Date deces"]))
    date_CR = datetime.datetime.strptime(date_CR, '%Y%m%d')
    date_death = datetime.datetime.strptime(date_death, '%Y%m%d')

    diff = (date_death - date_CR).days
    if diff < 0 :
        count_dates_inversed += 1
        return NaN
    else :
        return diff

def main(arg):
    if not os.path.isdir(arg.folder_to_save):
        os.mkdir(arg.folder_to_save)

    folder_to_save = arg.folder_to_save
    if arg.number_of_lines == 0 :
        df = pd.read_csv(arg.data_file)
    else :
        df = pd.read_csv(arg.data_file, nrows=arg.number_of_lines)
    
    df = df.dropna()

    # EHR by patient
    plt.figure(1)
    ehr_by_id = df["Noigr"].value_counts()
    ehr_by_id.plot.hist(bins=20)
    plt.title("EHR by patient")
    plt.xlabel("Nb of EHR")
    plt.savefig(os.path.join(folder_to_save, "ehr_by_id.jpg"))

    # Words by sentence / sentence by cr
    tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")
    preprocessor = EHRPreprocesser()
    df_count = df.apply(lambda x : count_sentence(x, tokenizer, preprocessor), axis=1)

    total_sentence_max = df_count["total_sentence"].max()
    max_value_desired = 150
    plt.figure(2)
    plt.xlim((0, min(total_sentence_max, max_value_desired)))
    df_count["total_sentence"].plot.hist(bins=100)
    plt.title("Sentences per EHR")
    plt.xlabel("Nb of sentences")
    plt.savefig(os.path.join(folder_to_save, "sentences_by_ehr.jpg"))

    plt.figure(3)
    max_mean_tokens = df_count["mean_tokens"].max()
    max_mean_tokens_desired = 200 
    plt.xlim((0, min(max_mean_tokens, max_mean_tokens_desired)))
    df_count["mean_tokens"].plot.hist(bins=100)
    plt.title("Mean tokens per sentence per EHR")
    plt.xlabel("Mean number of tokens")
    plt.savefig(os.path.join(folder_to_save, "mean_tokens.jpg"))

    plt.figure(4)
    max_tokens = df_count["nb_tokens"].max()
    nb_tokens_desired = 1500 
    plt.xlim((0, min(max_tokens, nb_tokens_desired)))
    df_count["nb_tokens"].plot.hist(bins=100)
    plt.title("Number of tokens per EHR")
    plt.xlabel("Total number of tokens")
    plt.savefig(os.path.join(folder_to_save, "nb_tokens.jpg"))

    # Distribution of survival time
    df["survival_time"] = df.apply(compute_survival_time, axis=1)

    survival_time = df["survival_time"].dropna().sort_values(ascending=False)
    beta = survival_time.mean()


    sample_from_geom = -np.sort(-np.random.exponential(beta, len(survival_time)))
    fig, ax = plt.subplots()
    ax.step(survival_time, list(range(len(survival_time))), label = "survival time")
    ax.step(sample_from_geom, list(range(len(survival_time))), label = "sample from exp law - beta = {}".format(round(beta,1)))
    ax.set_title('Distribution survival time')
    ax.set_xlabel('Survival time')
    ax.legend()
    plt.savefig(os.path.join(folder_to_save, "survival_time.jpg"))




if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_file", type=str, default="data/ehr/train.csv", 
        help="data file path")
    parser.add_argument("-f", "--folder_to_save", type=str, default="data/data_viz", 
        help="folder to save the figures")
    parser.add_argument("-nl", "--number_of_lines", type=int, default=0, 
        help="number of lines of the csv to read, 0 for all")

    main(parser.parse_args())
