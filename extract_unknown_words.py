import argparse
import json
import pandas as pd
from collections import Counter
import os
import re

from transformers import CamembertTokenizer

def in_camembert_voc(word, voc):
    return f'▁{word}' in voc

def main(path, n_unknown_words, chunksize, max_chunk):
    df_chunk = pd.read_csv(path, chunksize=chunksize)

    counter = Counter()

    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    voc = set(tokenizer.get_vocab().keys())

    for i, chunk in enumerate(df_chunk):
        if max_chunk and i > max_chunk:
            break
        for tweet in chunk.text:
            for word in re.findall(r'\w+', tweet):
                word = word.lower()
                if not in_camembert_voc(word, voc):
                    counter[word] += 1

    json_path = f"{os.path.split(path)[1]}_{n_unknown_words}_{max_chunk}.json"
    with open(os.path.join("medical_voc", json_path), 'w') as f:
        json.dump(counter.most_common(n_unknown_words), f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="french_tweets_short.csv", 
        help="path to dataset")
    parser.add_argument("-n", "--n_unknown_words", type=int, default=1000,
        help="save the n most frequent unknown words")
    parser.add_argument("-c", "--chunksize", type=int, default=10,
        help="chunksize when reading the dataset")
    parser.add_argument("-m", "--max_chunk", type=int, default=0,
        help="stops when reached max_chunk. 0 does not stop")
    args = parser.parse_args()
    print(f"\n> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")
    
    main(args.path, args.n_unknown_words, args.chunksize, args.max_chunk)