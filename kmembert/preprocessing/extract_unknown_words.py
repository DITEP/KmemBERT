'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import argparse
import json
import pandas as pd
from collections import Counter
import os
import re
from tqdm import tqdm
from transformers import CamembertTokenizer

from ..utils import get_root

def in_camembert_voc(word, voc):
    return f'▁{word}' in voc

def main(args):
    """
    Extracts words that camembert doesn't know and creates a json file with the most frequent.

    Inputs: please refer bellow, to the argparse arguments.
    """
    # Get path root of the project
    path_root = get_root()
    # Get path root to the dataset
    path_dataset = os.path.join(path_root, "kmembert\KmemBERT", args.data_folder, "train.csv")
    # Read the current dataframe
    df_chunk = pd.read_csv(path_dataset)

    counter = Counter()
    # Load tokenizer
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    # Get vocab (set type) linked to this tokenizer, 32004 words in the voc
    voc = set(tokenizer.get_vocab().keys())

    # For each column of the dataframe (tqdm is a progress bar)
    for i, chunk in enumerate(tqdm(df_chunk)): # i: index of the column, chunk: name of the column
        if args.max_chunk and i >= args.max_chunk:
            break
        for ehr in chunk.Texte:
            for word in re.findall(r'\w+', ehr):
                word = word.lower()
                if not in_camembert_voc(word, voc):
                    counter[word] += 1

    counter = counter.most_common(args.n_unknown_words)

    json_path = f"{args.data_folder}_{args.n_unknown_words}_{args.max_chunk}.json"
    with open(os.path.join("medical_voc", json_path), 'w') as f:
        json.dump(counter, f, indent=4, ensure_ascii=False)

def get_args_extract_unknown_words():
    """
    Returns:
        args: argparse: to execute files in jupyter notebook
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
        help="data folder name")
    parser.add_argument("-n", "--n_unknown_words", type=int, default=1000,
        help="save the n most frequent unknown words")
    parser.add_argument("-c", "--chunksize", type=int, default=10,
        help="chunksize when reading the dataset")
    parser.add_argument("-m", "--max_chunk", type=int, default=0,
        help="stops when reached max_chunk. 0 does not stop")
    args = parser.parse_args("")

    return args

def get_extract_unknow_words(data: pd.DataFrame, column_name: str, out_name: str, args = get_args_extract_unknown_words()):
    """
    Extracts words that camembert doesn'tknow and creates a json file with the most frequent, but as a function.s
    Inputs: 
        data: pd.DataFrame to extract vocabulary
        column_name: name of the column of 'data' with sequences
        args: argsParse (please see in __main__ an example)
        out_name: name of the json file with vocabulary
    Do: 
        Writes a json file whose name is 'out_name'
    Output:
        None
    """
    # Get path root of the project
    path_root = get_root()

    counter = Counter()
    # Load tokenizer
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    # Get vocab (set type) linked to this tokenizer, 32004 words in the voc
    voc = set(tokenizer.get_vocab().keys())

    # For each sequence, count occurences of each word
    for i in range(0, data.shape[0]):
        for word in re.findall(r'\w+', data.iloc[i][column_name]):
            word = word.lower()
            if not in_camembert_voc(word, voc):
                counter[word] += 1

    counter = counter.most_common(args.n_unknown_words)

    json_path = f"{out_name}.json"
    with open(os.path.join(path_root, "kmembert\\KmemBERT", "medical_voc", json_path), 'w') as f:
        json.dump(counter, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
        help="data folder name")
    parser.add_argument("-n", "--n_unknown_words", type=int, default=1000,
        help="save the n most frequent unknown words")
    parser.add_argument("-c", "--chunksize", type=int, default=10,
        help="chunksize when reading the dataset")
    parser.add_argument("-m", "--max_chunk", type=int, default=0,
        help="stops when reached max_chunk. 0 does not stop")
    args = parser.parse_args()
    print(f"\n> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")
    
    main(args)