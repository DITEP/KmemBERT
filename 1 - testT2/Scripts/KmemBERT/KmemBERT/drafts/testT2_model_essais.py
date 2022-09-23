'''
    Author: Théo Di Piazza (CLB), CentraleSupelec
    Year: 2022
    
    Permits to test the kmembert-T2 and save all results.
    Please, make sure to have a dataset 'test.csv' in result folder.
    Please, make sure the data is in the right format. Example in 'test_format.csv'
    Please, make sure this file is executed in KmemBERT folder (HealthBERT project).

    command line: python testT2_model.py
    python3 testT2_model_essais.py
'''

##########################################################
# Import libraries and packages
from kmembert.utils import Config
from kmembert.models import TransformerAggregator
from kmembert.dataset import PredictionsDataset
from torch.utils.data import DataLoader
from collections import defaultdict
from kmembert.dataset import PredictionsDataset
from kmembert.models import TransformerAggregator
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve, r2_score, mean_squared_error
from kmembert.utils import pretty_time, printc, create_session, save_json, get_label_threshold, get_error, time_survival_to_label, collate_fn
from kmembert.utils import create_session, get_label_threshold, collate_fn
from sklearn.metrics import confusion_matrix
from time import time
from scipy import stats

import pandas as pd
import seaborn as sns
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import torch
import sys
##########################################################

##########################################################
# Load the pre-trained model
resume = "kmembert-T2"
config = Config()
config.resume = resume

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters of the pre-trained T2 model
nhead, num_layers, out_dim, time_dim = 8, 4, 2, 8

# Init model
model = TransformerAggregator(device, config, nhead, num_layers, out_dim, time_dim)

# Load the model
model.resume(config)
##########################################################

##########################################################
# argparse part to test the model
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
    help="data folder name")
parser.add_argument("-a", "--aggregator", type=str, default="transformer", 
    help="aggregator name", choices=['conflation', 'sanity_check', 'sanity_check_transformer', 'transformer'])
parser.add_argument("-r", "--resume", type=str, default = "kmembert-base", 
    help="result folder in which the saved checkpoint will be reused")
parser.add_argument("-e", "--epochs", type=int, default=2, 
    help="number of epochs")
parser.add_argument("-nr", "--nrows", type=int, default=None, 
    help="maximum number of samples for training and validation")
parser.add_argument("-k", "--print_every_k_batch", type=int, default=1, 
    help="prints training loss every k batch")
parser.add_argument("-dt", "--days_threshold", type=int, default=365, 
    help="days threshold to convert into classification task")
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, 
    help="model learning rate")
parser.add_argument("-wg", "--weight_decay", type=float, default=0, 
    help="the weight decay for L2 regularization")
parser.add_argument("-p", "--patience", type=int, default=4, 
    help="number of decreasing accuracy epochs to stop the training")
parser.add_argument("-me", "--max_ehrs", type=int, default=4, 
    help="maximum nusmber of ehrs to be used for multi ehrs prediction")
parser.add_argument("-nh", "--nhead", type=int, default=8, 
    help="number of transformer heads")
parser.add_argument("-nl", "--num_layers", type=int, default=4, 
    help="number of transformer layers")
parser.add_argument("-od", "--out_dim", type=int, default=2, 
    help="transformer out_dim (1 regression or 2 density)")
parser.add_argument("-td", "--time_dim", type=int, default=8, 
    help="transformer time_dim")
args = parser.parse_args("")
##########################################################

##########################################################
# Load data and test the model
path_dataset, path_result, device, config = create_session(args)

print("path_result :", path_result)

assert (768 + args.time_dim) % args.nhead == 0, f'd_model (i.e. 768 + time_dim) must be divisible by nhead. Found time_dim {args.time_dim} and nhead {args.nhead}'

config.label_threshold = get_label_threshold(config, path_dataset)

dataset = PredictionsDataset(path_dataset, config, output_hidden_states=True, device=device, train=False)
loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

model.config.mode = "density"

test_losses = []
validation = True
model.eval()
predictions, test_labels, stds, noigr = [], [], [], []
test_start_time = time()

# Ajout THEO pour tracabilite prediction et interpreter
indice_theo, test_labels_theo, pred_theo = [], [], []

total_loss = 0
for i, (*data, labels, row) in enumerate(loader):
    noigr.append(id)
    print("len data:", len(data))
    # Ajout THEO pour tracabilite prediction et interpretation
    test_labels_theo.append(labels.item())
    indice_theo.append(int(row.item()))

    data = [[data[0][0]], torch.tensor(data[0][1], dtype=torch.float32)] # Transformation to adjust
    loss, outputs = model.step(*data, labels)
    
    #pred_theo.append(outputs.item())

    if model.mode == 'classif':
        predictions += torch.softmax(outputs, dim=1).argmax(axis=1).tolist()
    elif model.mode == 'regression':
        predictions += outputs.flatten().tolist()
    elif model.mode == 'density':
        mus, log_vars = outputs
        predictions += mus.tolist()
        stds += torch.exp(log_vars/2).tolist()
    elif model.mode == 'multi':
        if model.config.mode == 'density' or (model.config.mode == 'classif' and model.out_dim == 2):
            mu, log_var = outputs
            predictions.append(mu.item())
            stds.append(torch.exp(log_var/2).item())
            pred_theo.append(mu.item())
        else:
            predictions.append(outputs.item())
    else:
        raise ValueError(f'Mode {model.mode} unknown')
    
    #test_labels += labels.tolist()
    total_loss += loss.item()

    print("label, pred, row:", row.item(), mu.item(), labels.item())

mean_loss = total_loss/(config.batch_size*len(loader))

if test_losses is not None:
    test_losses.append(mean_loss)

#error = get_error(test_labels, predictions, config.mean_time_survival)
#printc(f"    {'Validation' if validation else 'Test'} | MAE: {int(error)} days - Global average loss: {mean_loss:.6f} - Time elapsed: {pretty_time(time()-test_start_time)}\n", 'RESULTS')


print('    Saving predictions, labels and index...')

predictions = np.array(predictions)
test_labels = np.array(test_labels)

# Save prediction as df
resu_pred = pd.DataFrame({"pred":pred_theo,
                        "lab":test_labels_theo,
                        "ind":indice_theo})
resu_pred.to_csv(path_result+"/tracabilite_prediction.csv", index=False)

print(f"    (Ended {'validation' if validation else 'testing'})\n")
##########################################################

# END
##########################################################