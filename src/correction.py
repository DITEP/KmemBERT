import argparse
import json
import pandas as pd
import os
"""
pip install pyspellchecker
pip install spacy
python -m spacy download fr
python -m spacy download fr_core_news_md
pip install pandarallel
"""
from spellchecker import SpellChecker
import spacy
from pandarallel import pandarallel

from utils import get_root

import pkg_resources
from symspellpy import SymSpell, Verbosity
from time import time

nlp = spacy.load('fr')
pandarallel.initialize(progress_bar=True)

def main(dataset, distance):
    """
    Corrects a dataset with SpellChecker and creates a new one.

    Inputs: please refer bellow, to the argparse arguments.
    """

    spell = SpellChecker(language='fr',distance=distance)

    def transform_one_sentence(sentence, method='SpellChecker'):
        """
        Transforms a sentence as a string into another setence with the unknown words being corrected
        # """

        if method=='SymSpell':
            sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            sym_spell.load_dictionary('../data/fr-100k.txt', term_index=0, count_index=1)
            suggestions = sym_spell.lookup_compound(sentence, max_edit_distance=2)
            return suggestions[0].term
            
        elif method=='SpellChecker':
            words = list(map(str, nlp(sentence)))
            return ' '.join(list(map(spell.correction, words)))

    path_root = get_root()
    path_dataset = os.path.join(path_root, "data", args.dataset)
    df = pd.read_csv(path_dataset)
<<<<<<< HEAD
    df.text = df.text.parallel_apply(transform_one_sentence, method= args.method)
=======
    df["Texte"] = df["Texte"].parallel_apply(transform_one_sentence)
>>>>>>> 130c271cf6641ad94d8f94254de81e49fced46e0
    
    correction_dataset = f'_correction_{distance}.csv'.join(path_dataset.split(".csv"))
    print(f"Saving the corrected dataset into {correction_dataset}...")
    df.to_csv(correction_dataset, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="../data/french_tweets_short.csv", 
        help="dataset filename")
    parser.add_argument("-n", "--distance", type=int, default=2, 
        help="distance parameter for SpellChecker")
    parser.add_argument("-m", "--method", type=str, default='SpellChecker', 
        help="Method of correction")
    args = parser.parse_args()
    print(f"\n> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")
    main(args.dataset, args.distance)