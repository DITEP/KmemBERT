import argparse
import json
import pandas as pd
import os
from pandarallel import pandarallel
from symspellpy import SymSpell
from tqdm import tqdm
tqdm.pandas()
pandarallel.initialize(progress_bar=True)

from utils import get_root, printc

def main(args):
    """
    Corrects a dataset with SymSpell and creates a new one.

    Inputs: please refer bellow, to the argparse arguments.
    """

    sym_spell = SymSpell()
    sym_spell.load_dictionary(args.dict_path, term_index=0, count_index=1)

    corrector = lambda sentence: sym_spell.lookup_compound(sentence.lower(), max_edit_distance=2, ignore_non_words=True)[0].term

    path_root = get_root()
    path_dataset = os.path.join(path_root, "data", args.data_folder)
    path_dataset_corrected = os.path.join(path_root, "data", f"{args.data_folder}_corrected_{args.distance}")

    os.mkdir(path_dataset_corrected)

    for csv_name in ['train.csv', 'test.csv']:
        path_csv = os.path.join(path_dataset, csv_name)
        path_corrected_csv = os.path.join(path_dataset_corrected, csv_name)
        print(f"Correcting {path_csv}...")

        df = pd.read_csv(path_csv)
        if args.parallel_apply:
            df["Texte"] = df["Texte"].parallel_apply(corrector)
        else:
            df["Texte"] = df["Texte"].progress_apply(corrector)

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
    parser.add_argument("-pa", "--parallel_apply", type=bool, default=False, const=True, nargs="?", 
        help="use parallel_apply")
    
    args = parser.parse_args()
    print(f"\n> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")
    main(args)