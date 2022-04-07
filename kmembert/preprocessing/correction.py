'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import argparse
import json
import pandas as pd
import os
from symspellpy import SymSpell, Verbosity
from tqdm import tqdm
tqdm.pandas()
import spacy
import shutil

from ..utils import printc
from ..preprocesser import EHRPreprocesser

class TextCorrector:
    def __init__(self, nlp, corrector, min_token_length = 5):
        self.nlp = nlp # corpus of text (ex: spacy fr)
        self.corrector = corrector # function which returns corrector
        self.min_token_length = min_token_length
        self.preprocesser = EHRPreprocesser()
        self.reset()

    def reset(self):
        self.num_tokens = 0
        self.num_corrected_tokens = 0
        self.num_long_tokens = 0
        self.correction_failed = 0

    def capitalize(self, tokens):
        tokens[0] = tokens[0].capitalize()
        for i in range(len(tokens)-1):
            if tokens[i] and tokens[i][0] == '.':
                tokens[i+1] = tokens[i+1].capitalize()

    def __call__(self, text):
        try:
            tokens = []
            text = self.preprocesser(text.lower()).strip()
            text = ' '.join(text.split())
            for token in self.nlp(text):
                self.num_tokens += 1
                if len(token) < self.min_token_length:
                    tokens.append(token.text_with_ws)
                else:
                    self.num_long_tokens += 1
                    tokens.append(self.corrector(str(token)))
                    if tokens[-1] != str(token):
                        self.num_corrected_tokens += 1
                    tokens.append(token.whitespace_)

            self.capitalize(tokens)
            return ''.join(tokens)
        except:
            print('CORRECTION FAILED')
            self.correction_failed += 1
            return text

def get_args_corrector():
    """
    Returns:
        argparse to execute get_corrector in other function in local
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
        help="data folder name")
    parser.add_argument("-dict", "--dict_name", type=str, default="fr-100k.txt", 
        help="french dict path. Please use the default one except if you know what you are doing")
    parser.add_argument("-dist", "--distance", type=int, default=2, 
        help="distance parameter")
    parser.add_argument("-mtl", "--min_token_length", type=int, default=5, 
        help="min token length to be corrected")
    
    args = parser.parse_args("")

    return args

def get_corrector(args):
    sym_spell = SymSpell()
    #dict_path = os.path.join('medical_voc', args.dict_name)
    dict_path = os.path.join('medical_voc', "fr-100k.txt")
    sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
    return lambda word: sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=args.distance, include_unknown=True)[0].term

def main(args):
    """
    Corrects a dataset with SymSpell and creates a new one.

    Inputs: please refer bellow, to the argparse arguments.
    """
    
    nlp = spacy.load('fr_core_news_sm')


    corrector = get_corrector(args)

    text_corrector = TextCorrector(nlp, corrector, min_token_length=args.min_token_length)

    path_dataset = os.path.join("data", args.data_folder)
    path_dataset_corrected = os.path.join("data", f"{args.data_folder}_corrected_{args.distance}")

    os.mkdir(path_dataset_corrected)
    shutil.copyfile(os.path.join(path_dataset, 'config.json'), 
                    os.path.join(path_dataset_corrected, 'config.json'))
    shutil.copyfile(os.path.join(path_dataset, 'validation_split.csv'), 
                    os.path.join(path_dataset_corrected, 'validation_split.csv'))

    for csv_name in ['train.csv', 'test.csv']:
        path_csv = os.path.join(path_dataset, csv_name)
        path_corrected_csv = os.path.join(path_dataset_corrected, csv_name)
        print(f"Correcting {path_csv}...")

        df = pd.read_csv(path_csv)
        text_corrector.reset()

        df["Texte"] = df["Texte"].progress_apply(text_corrector)
        
        print('num_corrected_tokens:', text_corrector.num_corrected_tokens)
        print(f"{100*text_corrector.num_corrected_tokens/text_corrector.num_tokens:.2f}% of all / {100*text_corrector.num_corrected_tokens/text_corrector.num_long_tokens:.2f}% of long tokens")
        print('correction_failed:', text_corrector.correction_failed, f"({100*text_corrector.correction_failed/len(df):.2f}%)")

        df.to_csv(path_corrected_csv, index=False)
        printc(f" > Corrected csv saved into {path_corrected_csv}\n", 'SUCCESS')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
        help="data folder name")
    parser.add_argument("-dict", "--dict_name", type=str, default="fr-100k.txt", 
        help="french dict path. Please use the default one except if you know what you are doing")
    parser.add_argument("-dist", "--distance", type=int, default=2, 
        help="distance parameter")
    parser.add_argument("-mtl", "--min_token_length", type=int, default=5, 
        help="min token length to be corrected")
    
    args = parser.parse_args()
    print(f"\n> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")
    main(args)
