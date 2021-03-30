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
import json
import numpy as np
from transformers import CamembertTokenizerFast

from ..preprocesser import EHRPreprocesser
from ..utils import print_args, get_label
from .extract_unknown_words import in_camembert_voc

def count_sentence(x, tokenizer, preprocesser, voc):
    count_token = 0

    x["Texte"] = preprocesser(x["Texte"])

    known_tokens = 0
    for word in re.findall(r'\w+', x["Texte"]):
        word = word.lower()
        if in_camembert_voc(word, voc):
            known_tokens += 1

    nb_tokens = len(tokenizer(x["Texte"], max_length=2000)["input_ids"])
    splitted_by_sentence = re.split(r'[.!?]+', str(x["Texte"]))
    total_sentence = len(splitted_by_sentence) - 1
    for s in splitted_by_sentence :
        if len(s): count_token += (len(tokenizer(s)["input_ids"]) - 1)

    mean_tokens = count_token / max(1, total_sentence)
    return pd.Series([total_sentence, mean_tokens, nb_tokens, known_tokens], index = ["total_sentence", "mean_tokens", "nb_tokens", "known_tokens"])


def main(args):
    print_args(args)

    folder_to_save = os.path.join(args.folder_to_save, args.data_folder)
    if not os.path.isdir(folder_to_save):
        os.mkdir(folder_to_save)

    path_dataset = os.path.join("data", args.data_folder, 'train.csv')
    if not args.number_of_lines:
        df = pd.read_csv(path_dataset)
    else:
        df = pd.read_csv(path_dataset, nrows=args.number_of_lines)
    df = df.dropna()


    # EHR by patient
    plt.figure(1)
    ehr_by_id = df["Noigr"].value_counts()
    ehr_by_id.plot.hist(bins=20)
    plt.title("EHR by patient")
    plt.xlabel("Nb of EHR")
    plt.savefig(os.path.join(folder_to_save, "ehr_distribution.png"))
    plt.close()


    # Words by sentence / sentence by cr
    tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")
    preprocessor = EHRPreprocesser()

    if args.voc_file:
        print(f'Using vocabulary of {args.voc_file}')
        with open(os.path.join('medical_voc', args.voc_file)) as json_file:
            voc_list = json.load(json_file)
            tokenizer.add_tokens([ token for (token, _) in voc_list ])
    voc = set(tokenizer.get_vocab().keys())

    df_count = df.apply(lambda x : count_sentence(x, tokenizer, preprocessor, voc), axis=1)

    total_sentence_max = df_count["total_sentence"].max()
    max_value_desired = 150
    plt.figure(2)
    plt.xlim((0, min(total_sentence_max, max_value_desired)))
    df_count["total_sentence"].plot.hist(bins=100)
    plt.title("Sentences per EHR")
    plt.xlabel("Nb of sentences")
    plt.savefig(os.path.join(folder_to_save, "sentences_distribution.png"))
    plt.close()

    plt.figure(3)
    max_mean_tokens = df_count["mean_tokens"].max()
    max_mean_tokens_desired = 200 
    plt.xlim((0, min(max_mean_tokens, max_mean_tokens_desired)))
    df_count["mean_tokens"].plot.hist(bins=100)
    plt.title("Mean tokens per sentence per EHR")
    plt.xlabel("Mean number of tokens")
    plt.savefig(os.path.join(folder_to_save, "mean_tokens_distribution.png"))
    plt.close()

    plt.figure(4)
    max_tokens = df_count["nb_tokens"].max()
    nb_tokens_desired = 1500 
    plt.xlim((0, min(max_tokens, nb_tokens_desired)))
    df_count["nb_tokens"].plot.hist(bins=100)
    plt.title("Number of tokens per EHR")
    plt.xlabel("Total number of tokens")
    plt.savefig(os.path.join(folder_to_save, "nb_tokens_distribution.png"))
    plt.close()

    plt.figure(6)
    max_known_tokens = df_count["known_tokens"].max()
    nb_tokens_desired = 1500 
    plt.xlim((0, min(max_known_tokens, nb_tokens_desired)))
    df_count["known_tokens"].plot.hist(bins=100)
    plt.title("Number of known tokens per EHR")
    plt.xlabel("Total number of tokens")
    plt.savefig(os.path.join(folder_to_save, f"nb_known_tokens_distribution_{args.voc_file if args.voc_file else 'no-voc'}.png"))
    plt.close()


    # Distribution of survival time
    survival_times = np.array(list(df[["Date deces", "Date cr"]].apply(lambda x: get_label(*x), axis=1)))
    survival_times[::-1].sort()
    beta = survival_times.mean()

    sample_from_geom = -np.sort(-np.random.exponential(beta, len(survival_times)))
    _, ax = plt.subplots()
    ax.step(survival_times, list(range(len(survival_times))), label = "survival time")
    ax.step(sample_from_geom, list(range(len(survival_times))), label = "sample from exp law - beta = {}".format(round(beta,1)))
    ax.set_title('Distribution survival time')
    ax.set_xlabel('Survival time (days)')
    ax.legend()
    plt.savefig(os.path.join(folder_to_save, "survival_time_distribution.png"))
    plt.close()




if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
        help="data folder name")
    parser.add_argument("-f", "--folder_to_save", type=str, default="graphs", 
        help="folder to save the figures")
    parser.add_argument("-v", "--voc_file", type=str, default=None, 
        help="file to the new words to be added to the tokenizer")
    parser.add_argument("-nl", "--number_of_lines", type=int, default=None, 
        help="number of lines of the csv to read, 0 for all")

    main(parser.parse_args())
