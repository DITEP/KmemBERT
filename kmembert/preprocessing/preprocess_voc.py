'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import json
import argparse
import os

from ..utils import printc
from .correction import get_corrector, get_args_corrector

from ..utils import get_root

def main(args):
    n_duplicate, n_misspelled = 0, 0

    corrector = get_corrector(args)

    with open(os.path.join('medical_voc', args.voc_file)) as json_file:
        voc_list = json.load(json_file)
        voc = set(x for x,_ in voc_list)

    new_list = []
    for (word, occ) in voc_list: # For every word in the vocabulary
        if word[-1] == 's' and word[:-1] in voc: # If last letter is 's' + word without 's' already exsists : its the same
            n_duplicate += 1
            continue
        
        corrected_word = corrector(word) # Every word in the vocab is corrected
        if len(word) >= args.min_token_length and corrected_word != word: # If the word is indeed corrected => correction added to the new list
            new_list.append([corrected_word, occ])
            n_misspelled += 1
        else: # If the word is well spelled => added to the new list
            new_list.append([word, occ])
        
    print(f"Removed {n_duplicate}/{len(voc)} duplicates (same word with an s)")
    print(f"Corrected {n_misspelled}/{len(voc)} missplelled words")

    with open(os.path.join('medical_voc', f"p_{args.voc_file}"), 'w') as f:
        json.dump(new_list, f, indent=4, ensure_ascii=False)
    printc("Successfully preprocess and saved", "SUCCESS")
    
def get_args_preprocess_voc():
    """
    Function to create argparse and returns it (used for local execution in jupyter notebook)
    Returns:
        argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--voc_file", type=str, default="toto1",
        help='voc file name')
    parser.add_argument("-dict", "--dict_name", type=str, default="fr-100k.txt", 
        help="dict path")
    parser.add_argument("-dist", "--distance", type=int, default=2, 
        help="distance parameter")
    parser.add_argument("-mtl", "--min_token_length", type=int, default=5, 
        help="min token length to be corrected")
    
    args = parser.parse_args("")

    return(args)

def get_preprocess_voc(voc_json_name: str, args = get_args_preprocess_voc()):
    """
    From vocab extracted from df, correct misspelled words or add new ones (so errors among new ones)
    Inputs:
        voc_json_name: str: name of the json file (without .json extansion)
    """

    n_duplicate, n_misspelled = 0, 0

    corrector = get_corrector(get_args_corrector())

    # Get roots to load the dataset
    path_root = get_root() # Root of the project
    json_path = f"{voc_json_name}.json" # Name of the file as json
    with open(os.path.join(path_root, "kmembert\\KmemBERT", "medical_voc", json_path)) as json_file:
        voc_list = json.load(json_file)
        voc = set(x for x,_ in voc_list)

    new_list = []
    for (word, occ) in voc_list: # For every word in the vocabulary
        if word[-1] == 's' and word[:-1] in voc: # If last letter is 's' + word without 's' already exsists : its the same
            n_duplicate += 1
            continue
        corrected_word = corrector(word) # Every word in the vocab is corrected
        if len(word) >= args.min_token_length and corrected_word != word: # If the word is indeed corrected => correction added to the new list
            new_list.append([corrected_word, occ])
            n_misspelled += 1
        else: # If the word is well spelled => added to the new list
            new_list.append([word, occ])
        
    print(f"Removed {n_duplicate}/{len(voc)} duplicates (same word with an s)")
    print(f"Corrected {n_misspelled}/{len(voc)} missplelled words")

    with open(os.path.join(path_root, "kmembert\\KmemBERT", "medical_voc", voc_json_name + "_new.json"), 'w') as f:
        json.dump(new_list, f, indent=4, ensure_ascii=False)
    printc("Successfully preprocess and saved", "SUCCESS")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--voc_file", type=str, required=True,
        help='voc file name')
    parser.add_argument("-dict", "--dict_name", type=str, default="fr-100k.txt", 
        help="dict path")
    parser.add_argument("-dist", "--distance", type=int, default=2, 
        help="distance parameter")
    parser.add_argument("-mtl", "--min_token_length", type=int, default=5, 
        help="min token length to be corrected")
    
    args = parser.parse_args()
    print(f"\n> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")
    main(args)