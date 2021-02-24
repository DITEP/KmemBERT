import argparse
import json
import pandas as pd
from collections import Counter
import os
import re
from tqdm import tqdm

from transformers import CamembertTokenizer
from utils import get_root

def in_camembert_voc(word, voc):
    return f'▁{word}' in voc

def main(args):
    """
    Extracts words that camembert doesn't know and creates a json file with the most frequent.

    Inputs: please refer bellow, to the argparse arguments.
    """
    path_root = get_root()
    path_dataset = os.path.join(path_root, "data", args.data_folder, "train.csv")
    df_chunk = pd.read_csv(path_dataset, chunksize=args.chunksize)

    counter = Counter()

    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    voc = set(tokenizer.get_vocab().keys())

    for i, chunk in enumerate(tqdm(df_chunk)):
        if args.max_chunk and i > args.max_chunk:
            break
        for ehr in chunk.Texte:
            for word in re.findall(r'\w+', ehr):
                word = word.lower()
                if not in_camembert_voc(word, voc):
                    counter[word] += 1

    counter = counter.most_common(args.n_unknown_words)

    json_path = f"{args.data_folder}_{args.n_unknown_words}_{args.max_chunk}.json"
    with open(os.path.join("medical_voc", json_path), 'w') as f:
        json.dump(counter, f, indent=4)

    txt_path = f"{args.data_folder}_{args.n_unknown_words}_{args.max_chunk}.txt"
    with open(os.path.join("sentence_piece", txt_path), "w") as text_file:
        min_occurence = counter[-1][1]
        text_file.write(' '.join([ word for word, occurence in counter for _ in range(occurence // min_occurence)]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
        help="dataset filename")
    parser.add_argument("-n", "--n_unknown_words", type=int, default=1000,
        help="save the n most frequent unknown words")
    parser.add_argument("-c", "--chunksize", type=int, default=10,
        help="chunksize when reading the dataset")
    parser.add_argument("-m", "--max_chunk", type=int, default=0,
        help="stops when reached max_chunk. 0 does not stop")
    args = parser.parse_args()
    print(f"\n> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")
    
    main(args)