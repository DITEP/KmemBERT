import json
import argparse
import os

from ..utils import printc
from .correction import get_corrector

def main(args):
    n_duplicate, n_misspelled = 0, 0

    corrector = get_corrector(args)

    with open(os.path.join('medical_voc', args.voc_file)) as json_file:
        voc_list = json.load(json_file)
        voc = set(x for x,_ in voc_list)

    new_list = []
    for (word, occ) in voc_list:
        if word[-1] == 's' and word[:-1] in voc:
            n_duplicate += 1
            continue
        
        corrected_word = corrector(word)
        if len(word) >= args.min_token_length and corrected_word != word:
            new_list.append([corrected_word, occ])
            n_misspelled += 1
        else:
            new_list.append([word, occ])
        
    print(f"Removed {n_duplicate}/{len(voc)} duplicates (same word with an s)")
    print(f"Corrected {n_misspelled}/{len(voc)} missplelled words")

    with open(os.path.join('medical_voc', f"p_{args.voc_file}"), 'w') as f:
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