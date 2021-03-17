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

def count_sentence(x):
    total_sentence = 0
    count_word = 0
    splitted_by_sentence = re.split(r'[.!?]+', str(x["Texte"]))
    total_sentence += len(splitted_by_sentence) - 1
    for s in splitted_by_sentence :
        if len(s) == 0 :
            continue 
        count_word += len(s.split())
    mean_word = count_word / max(1, total_sentence)
    return pd.Series([total_sentence, mean_word], index = ["total_sentence", "mean_word"])

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
    df_count = df.apply(count_sentence, axis=1)

    total_sentence_max = df_count["total_sentence"].max()
    max_value_desired = 150
    plt.figure(2)
    plt.xlim((0, min(total_sentence_max, max_value_desired)))
    df_count["total_sentence"].plot.hist(bins=100)
    plt.title("Sentences per EHR")
    plt.xlabel("Nb of sentences")
    plt.savefig(os.path.join(folder_to_save, "sentences_by_ehr.jpg"))

    plt.figure(3)
    words_max = df_count["mean_word"].max()
    words_max_desired = 200 
    plt.xlim((0, min(words_max, words_max_desired)))
    df_count["mean_word"].plot.hist(bins=100)
    plt.title("Mean words per sentence per EHR")
    plt.xlabel("Mean number of words")
    plt.savefig(os.path.join(folder_to_save, "mean_words.jpg"))

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
    parser.add_argument("-f", "--folder_to_save", type=str, default="data_viz", 
        help="folder to save the figures")
    parser.add_argument("-nl", "--number_of_lines", type=int, default=0, 
        help="number of lines of the csv to read, 0 for all")

    main(parser.parse_args())
