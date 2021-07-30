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


    model = HealthBERT(device, config)
    file_to_classify = pd.read_csv(config.data_folder, nrows = config.nrows)
    texts_to_classify = file_to_classify.loc[file_to_classify.Noigr == 2].Texte.values



   # print(texts_to_classify)


  
    word_attributes = []
    for ehr in texts_to_classify:

        cls_explainer = SequenceClassificationExplainer(
            model.camembert,
            model.tokenizer)
        word_attributions = cls_explainer(ehr)
        word_attributes+= [dict(word_attributions)]
    #print(word_attributes)
    #final_dict = dict(functools.reduce(operator.add, map(collections.Counter, word_attributes)))
    result = {}
    n_documents = len(word_attributes)
    for d in word_attributes:
        for k in d.keys():
            result[k] = result.get(k, 0) + d[k]/n_documents
    #print(result)
    final_dict = dict(sorted(result.items(), key=lambda item: abs(item[1]))[:10])
    #print(final_dict)
    x = list(final_dict.keys())
    y = list(final_dict.values())
    colors = ['red']*10
    for i in range(len(y)):
        if y[i]>0:
            colors[i] = 'green'
    plt.figure()
    plt.barh(x, y, color=colors)
    plt.title('Word Attributions for all EHRs of a selected patient')
    plt.savefig('graphs/interpretation/test1.png')    
    plt.close()
    #print(final_dict)




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













