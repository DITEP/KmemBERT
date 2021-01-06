import argparse
import json
import pandas as pd

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

spell = SpellChecker(language='fr')
nlp = spacy.load('fr')

pandarallel.initialize()

def transform_one_sentence(sentence):
    """
    Transforms a sentence as a string into another setence with the unknown words being corrected
    """
    words = list(map(str, nlp(sentence)))
    return ' '.join(list(map(spell.correction, words)))


def main(dataset):
    df = pd.read_csv(dataset)
    df.text = df.text.parallel_apply(transform_one_sentence)

    correction_dataset = '_correction.csv'.join(dataset.split(".csv"))
    print(f"Saving the corrected dataset into {correction_dataset}...")
    df.to_csv(correction_dataset, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="french_tweets_short.csv", 
        help="dataset filename")

    args = parser.parse_args()
    print(f"\n> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")
    main(args.dataset)