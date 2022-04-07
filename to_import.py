#!python -m spacy download fr_core_news_sm
import os
import sys
import argparse

def get_arg():

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