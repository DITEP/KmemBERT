import argparse
import json
import pandas as pd
import os
from pandarallel import pandarallel
from symspellpy import SymSpell, Verbosity
from tqdm import tqdm
import spacy
tqdm.pandas()

from utils import get_root, printc

class TextCorrector:
    def __init__(self, nlp, corrector, min_token_length = 5):
        self.nlp = nlp
        self.corrector = corrector
        self.min_token_length = min_token_length

    def capitalize(self, tokens):
        tokens[0] = tokens[0].capitalize()
        for i in range(len(tokens)-1):
            if tokens[i] == '.':
                tokens[i+1] = tokens[i+1].capitalize()

    def __call__(self, text):
        tokens = []
        text = text.lower()
        for token in self.nlp(text):
            if len(token) < self.min_token_length:
                tokens.append(str(token))
            else:
                tokens.append(self.corrector(str(token)))

        self.capitalize(tokens)
        return ' '.join(tokens)


def main(args):
    """
    Corrects a dataset with SymSpell and creates a new one.

    Inputs: please refer bellow, to the argparse arguments.
    """
    if args.parallel_apply:
        pandarallel.initialize(progress_bar=True)

    nlp = spacy.load('fr')

    sym_spell = SymSpell()
    sym_spell.load_dictionary(args.dict_path, term_index=0, count_index=1)
    corrector = lambda word: sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=args.distance, include_unknown=True)[0].term

    text_corrector = TextCorrector(nlp, corrector, min_token_length=args.min_token_length)

    path_root = get_root()
    path_dataset = os.path.join(path_root, "data", args.data_folder)
    path_dataset_corrected = os.path.join(path_root, "data", f"{args.data_folder}_corrected_{args.distance}")

    os.mkdir(path_dataset_corrected)

    for csv_name in ['train.csv', 'test.csv']:
        path_csv = os.path.join(path_dataset, csv_name)
        path_corrected_csv = os.path.join(path_dataset_corrected, csv_name)
        print(f"Correcting {path_csv}...")

        df = pd.read_csv(path_csv, sep='secrettoken749386453728394027', engine='python')
        if args.parallel_apply:
            df["Texte"] = df["Texte"].parallel_apply(text_corrector)
        else:
            df["Texte"] = df["Texte"].progress_apply(text_corrector)

        df.to_csv(path_corrected_csv, index=False)
        printc(f" > Corrected csv saved into {path_corrected_csv}\n", 'SUCCESS')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
        help="data folder name")
    parser.add_argument("-dict", "--dict_path", type=str, default="data/fr-100k.txt", 
        help="dict path")
    parser.add_argument("-dist", "--distance", type=int, default=2, 
        help="distance parameter")
    parser.add_argument("-mtl", "--min_token_length", type=int, default=5, 
        help="min token length to be corrected")
    parser.add_argument("-pa", "--parallel_apply", type=bool, default=False, const=True, nargs="?", 
        help="use parallel_apply")
    
    args = parser.parse_args()
    print(f"\n> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")
    main(args)