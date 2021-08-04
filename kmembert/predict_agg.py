'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os

from .models import HealthBERT
from .utils import create_session, get_date, get_label






def main(args):
    global model
    _, _, device, config = create_session(args)
    #config = Config(args)
    # config.path_result = ""
    # config.resume = "training_21-04-05_10h02m00s"

    ehrs = pd.read_csv(config.data_folder)
    ehrs['date_ehr'] = ehrs["Date cr"].apply(lambda x: get_date(str(x)))
    deces = ehrs['Date deces'].values
    cr = ehrs["Date cr"].values
    ehrs['label'] = [get_label(deces[i], cr[i]) for i in range(len(deces))]

    ehrs = ehrs.sort_values(by = ['Date cr'])
    ehrs = ehrs.iloc[-5:]


    ehrs.to_csv('data/ehr/test_sample.csv')



    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="data/ehr/train.csv", 
        help="data path to access to the testing file")
    parser.add_argument("-p", "--path_dataset", type=str, default="data/ehr/train.csv", 
        help="data path to access to the testing file")
    parser.add_argument("-r", "--resume", type=str, required=True, 
        help="result folder in with the saved checkpoint will be reused")
    
    parser.add_argument("-nr", "--nrows", type=int, default=10, 
        help="maximum number of samples for testing")
    parser.add_argument("-f", "--folder_to_save", type=str, default="graphs", 
        help="folder to save the figures")
    main(parser.parse_args())













