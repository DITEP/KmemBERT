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

def main(dataset, n_unknown_words, chunksize, max_chunk):
    """
    Extracts words that camembert doesn't know and creates a json file with the most frequent.

    Inputs: please refer bellow, to the argparse arguments.
    """
    path_root = get_root()
    path_dataset = os.path.join(path_root, "data", dataset)
    df_chunk = pd.read_csv(path_dataset, chunksize=chunksize)

    counter = Counter()

    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    voc = set(tokenizer.get_vocab().keys())

    for i, chunk in enumerate(tqdm(df_chunk)):
        if max_chunk and i > max_chunk:
            break
        for tweet in chunk.text:
            for word in re.findall(r'\w+', tweet):
                word = word.lower()
                if not in_camembert_voc(word, voc):
                    counter[word] += 1

    json_path = f"{os.path.split(path_dataset)[1]}_{n_unknown_words}_{max_chunk}.json"
    with open(os.path.join("medical_voc", json_path), 'w') as f:
        json.dump(counter.most_common(n_unknown_words), f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="french_tweets_short.csv", 
        help="dataset filename")
    parser.add_argument("-n", "--n_unknown_words", type=int, default=1000,
        help="save the n most frequent unknown words")
    parser.add_argument("-c", "--chunksize", type=int, default=10,
        help="chunksize when reading the dataset")
    parser.add_argument("-m", "--max_chunk", type=int, default=0,
        help="stops when reached max_chunk. 0 does not stop")
    args = parser.parse_args()
    print(f"\n> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")
    
    main(args.dataset, args.n_unknown_words, args.chunksize, args.max_chunk)