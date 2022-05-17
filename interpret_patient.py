'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import numpy as np
import pandas as pd
import argparse
from torch.serialization import save
from transformers_interpret import SequenceClassificationExplainer
import matplotlib.pyplot as plt
import os
import collections, functools, operator

from .models import HealthBERT
from .utils import create_session







def main(args):
    global model
    _, _, device, config = create_session(args)
    
    #config = Config(args)
    # config.path_result = ""
    # config.resume = "training_21-04-05_10h02m00s"
    noigr = args.noigr
    # First, we load the model (we will only use the camembert part)
    model = HealthBERT(device, config)
    # Then we load the texts to classify. We can load them in a different way. 
    # Here we chose to have the word attributes for a certain patient in some of their EHRs.
    file_to_classify = pd.read_csv(config.data_folder, nrows = config.nrows)
    texts_to_classify = file_to_classify.loc[file_to_classify.Noigr == noigr].Texte.values
    
    # We start the interpretation using transformers-interpret. 
    # Word attributes is a list of dictionary. Each dictionary corresponds to the word attributes of a text from texts_to_classify. 
    word_attributes = []
    for ehr in texts_to_classify:

        cls_explainer = SequenceClassificationExplainer(
            model.camembert,
            model.tokenizer)
        word_attributions = cls_explainer(ehr)
        word_attributes+= [dict(word_attributions)]
    
    #final_dict = dict(functools.reduce(operator.add, map(collections.Counter, word_attributes)))
    
    # We try to get each word contribution in all the documents of the same patient
    # Can be ignored if we want word attributes for each document.
    
    result = {}
    n_documents = len(word_attributes)
    for d in word_attributes:
        for k in d.keys():
            result[k] = result.get(k, 0) + d[k]/n_documents
    
    # Here we sort the words by their contribution to the output and take the first 10 (can be changed)
    show_n = 10
    final_dict = dict(sorted(result.items(), key=lambda item: abs(item[1]))[:show_n])
    
    # Plot the results for the word attributes (green for positive contribution and red for negative contribution)
    x = list(final_dict.keys())
    y = list(final_dict.values())
    colors = ['red']*show_n 
    for i in range(len(y)):
        if y[i]>0:
            colors[i] = 'green'
    plt.figure()
    plt.barh(x, y, color=colors)
    plt.title(f'Word Attributions for all EHRs of a the patient {noigr}')
    plt.savefig(f'graphs/interpretation/test_{noigr}.png')    
    plt.close()




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
    parser.add_argument("-ng", "--noigr", type=int, default=2, 
        help="The Noigr of a patient")
    main(parser.parse_args())













