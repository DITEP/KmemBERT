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
from scipy.special import softmax
from .models import HealthBERT
from .utils import create_session
import json


def show_attention(sentence):
    inputs = model.tokenizer.encode(sentence, return_tensors='pt')
    outputs = model.camembert(inputs, output_attentions=True)
    attention = outputs[-1]
    tokens = model.tokenizer.convert_ids_to_tokens(inputs[0])
    attention = np.array([l.detach().numpy() for l in attention])
    return tokens, attention


def compute_attention(tokens, attention):
    evolution = np.zeros((12, len(tokens)))
    for i in range(12):
        att = attention[i][0]
        for j in range(len(tokens)):
            evolution[i][j] = att[:,:, j].sum(-1).mean()
    return softmax(evolution, axis = 0)


def main(args):
    global model
    _, _, device, config = create_session(args)
    #config = Config(args)
    # config.path_result = ""
    # config.resume = "training_21-04-05_10h02m00s"


    model = HealthBERT(device, config)
    file_to_classify = pd.read_csv(config.data_folder, nrows = config.nrows)
    texts_to_classify = file_to_classify.Texte.values
    f = open("medical_voc/large.json")
    dictio = json.load(f)
    med_voc = []
    for i in range(len(dictio)):
        med_voc.append(dictio[i][0])
    
    input = model.tokenizer.encode(' '.join(med_voc), return_tensors='pt')
    medical_vocab = model.tokenizer.convert_ids_to_tokens(input[0])

 
    medical_vocab.remove('<s>')
    medical_vocab.remove('</s>')

    evolution = []
    for sentence in texts_to_classify:
        sentence_tokenized, attention = show_attention(sentence)
        idx_med = []
        for i in range(len(sentence_tokenized)):
            if (len(sentence_tokenized[i]) > 3) and sentence_tokenized[i] in medical_vocab:
                print(sentence_tokenized[i])
                idx_med.append(i)
        evolutionary = compute_attention(sentence_tokenized, attention)
        evolutions = evolutionary[:, idx_med].sum(axis = 1)
        evolution.append(evolutions)
    
    final_evolution = np.array(evolution).mean(axis = 0)
    print(final_evolution)
    plt.figure()
    plt.plot(final_evolution)
    plt.savefig('graphs/test2.png')
        



  
   
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













