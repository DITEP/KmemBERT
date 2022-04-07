'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, date
import sys
import json
import numpy as np
import matplotlib.pyplot as plt 

from .config import Config
from sklearn.metrics import confusion_matrix
from collections import defaultdict

def get_root():
    """
    Gets the absolute path to the root of the project
    """
    #return os.sep.join(os.getcwd().split(os.sep)[0 : os.getcwd().split(os.sep).index("EHR_Transformers") + 1])
    return os.sep.join(os.getcwd().split(os.sep)[0 : os.getcwd().split(os.sep).index("KmemBERT") + 1])

def pretty_time(t):
    """
    Tranforms time t in seconds into a pretty string
    """
    return f"{int(t//60)}m{int(t%60)}s"

def time_survival_to_label(time_survival, mean_time_survival):
    """
    Transforms times of survival into uniform labels in ]0,1[
    """
    return 1 - np.exp(-time_survival/mean_time_survival)

def label_to_time_survival(label, mean_time_survival):
    """
    Transforms labels in ]0,1[ into times of survival
    """
    return - mean_time_survival*np.log(
        np.clip(1-label, a_min=1e-5, a_max=1))

def shift_predictions(mus, mean_time_survival, shift):
    """
    Shift predictions in [0, 1] given a shift (transfer into survival time, 
    shift, and back into [0, 1])
    """
    time_survival = - mean_time_survival*torch.log(1-mus)
    time_survival -= shift
    return (1 - torch.exp(-time_survival/mean_time_survival)).clamp(0, 1)

bcolors = {
    'RESULTS': '\033[95m',
    'HEADER': '\033[94m',
    'SUCCESS': '\033[92m',
    'WARNING': '\033[93m',
    'FAIL': '\033[91m',
    'ENDC': '\033[0m',
    'INFO': '\033[1m',
    'UNDERLINE': '\033[4m'
}

def printc(log, color='HEADER'):
    """
    Prints logs with color according to the dict bcolors
    """
    print(f"{bcolors[color]}{log}{bcolors['ENDC']}")

def now():
    """
    Current date as a string
    """
    return datetime.now().strftime('%y-%m-%d_%Hh%Mm%Ss')

def save_json(path_result, name, x):
    """
    Saves x into path_result with the given name
    """
    with open(os.path.join(path_result, f'{name}.json'), 'w') as f:
        json.dump(x, f, indent=4)

def print_args(args):
    """
    Prints argparse arguments from the command line
    """
    print(f"> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")

def create_session(args):
    """
    Initializes a script session (set seed, get the path to the result folder, ...)
    """
    torch.manual_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    printc(f"> DEVICE:  {device}", "INFO")

    path_root = get_root()
    printc(f"> ROOT:    {path_root}", "INFO")

    path_dataset = os.path.join(path_root, "data", args.data_folder)

    main_file = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    session_id = f"{main_file}_{now()}"
    path_result = os.path.join(path_root, "results", session_id)
    os.mkdir(path_result)
    printc(f"> SESSION: {path_result}", "INFO")

    args.path_result = path_result
    config = Config(args)
    
    save_json(path_result, 'args', vars(args))

    return path_dataset, path_result, device, config

def get_date(str_date):
    """
    Being given the string 20160211 returns date(2016,2,11)
    """
    year = int(str_date[:4])
    month = int(str_date[4:6])
    day = int(str_date[6:8])
    return date(year, month, day)

def get_label(str_date_deces, str_date_cr):
    """
    Being given 2 strings like 20160201 and 20170318 returns the corresponding time difference in number of days.
    Date format: yyyymmdd
    """
    
    date_deces = get_date(str(str_date_deces))
    date_cr = get_date(str(str_date_cr))

    delta = date_deces - date_cr
    return delta.days

def get_label_threshold(config, path_dataset):
    """
    Converts a threshold as a number of days into a threshold as a label in [0, 1]
    """
    config_path = os.path.join(path_dataset, "config.json")
    assert os.path.isfile(config_path), 'Config file not existing, please create it first using split_dataset.py'
    with open(config_path) as json_file:
        mean_time_survival = json.load(json_file)["mean_time_survival"]

    return time_survival_to_label(config.days_threshold, mean_time_survival)

def get_error(labels, predictions, threshold=0.5):
    """
    Computes the accuracy from np.array with filtration on "-1"
    Inputs:
        labels, predictions: np.array of labels, and raw-score of the model
    Output:
        Error of prediction, a scalar
    """
    # Filter to remove "-1" labels
    labels_filtered = np.array([item for item in labels if item != -1])
    predictions_filtered = np.array([predictions[i] for i in range(len(predictions)) if labels[i] != -1])

    # Convert Raw-Scores to probabilities then labels
    prob = 1/(1 + np.exp(-predictions_filtered)) # To probabilities
    prob = 1*(np.array(prob) >= threshold) # To labels

    # Compute error to return
    error = 1 - np.mean(labels_filtered == prob)

    return error

def get_error_IT(labels, predictions, threshold=0.5):
    """
    Computes the accuracy from np.array
    Inputs:
        labels, predictions: list of list for labels and predictions
    Outputs: dict such as {"Time Interval Index": "error"}
    """
    error_IT = {}
    # For each loop: we compute error of predicion for a given Time Interval 
    for i in range(len(labels[0])):

        # Get labels and prediction for current IT "i"
        lab = np.array([item[i] for item in labels]) # Get label for IT "i"
        pred = np.array([item[i] for item in predictions]) # Get prediction for IT "i"

        # Filter to remove "-1" labels
        filtered_labels = np.array([item for item in lab if item != -1])
        filtered_predictions = np.array([pred[i] for i in range(len(pred)) if lab[i] != -1])

        # Convert Raw-Scores to probabilities then labels
        prob = 1/(1 + np.exp(-filtered_predictions)) # Raw-Scores to probabilities (prob)
        prob = 1*(np.array(prob) >= threshold) # Probabilities to prediction labels
        filtered_labels = np.array(filtered_labels.tolist())

        # Compute error to return
        error = 1 - np.mean(filtered_labels == prob) # Compute error for given IT
        error_IT.update( {i: error} ) # Add error with "IT index" (i) as key

    return error_IT 

def collate_fn(batch):
    """
    Custom collate function for Predictions Dataset
    """
    *outputs, dt, label = batch[0]
    return (outputs, torch.tensor(dt).type(torch.float32), torch.tensor([label]).type(torch.float32))
   
def collate_fn_with_id(batch):
    """
    Custom collate function for Predictions Dataset
    """
    noigr, *outputs, dt, label = batch[0]
    return (noigr, outputs, torch.tensor(dt).type(torch.float32), torch.tensor([label]).type(torch.float32))

# Function: Plot loss as a function of epoch
def plot_epoch_loss(dico_train: dict, list_test: list, path_result):
    """
    Saves .png image: Loss as a function of epoch
    Inputs:
        dico_train: a dict with epochs as keys and loss (of train) as values
        list_test: a list with loss (of test)
        path_result: path to save the plot as png
    """
    # Rearrange data to plot from dict
    lists = sorted(dico_train.items()) # sorted by key, return a list of tuples
    x_train, y_train = zip(*lists) # unpack a list of pairs into two tuples
    y_test = list_test
    # Plot data
    plt.plot(x_train, y_train, label="Train")
    plt.plot(x_train, y_test, label = "Test")
    plt.title("Model Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(path_result, "loss.png"))
    plt.close()

    return None

# Function: Plot error as a function of epoch
def plot_epoch_error(dico_error_train: dict, dico_error_test: dict, path_result):
    """
    Saves .png image: Error as a function of epoch
    Inputs:
        dico_error: a dict with epochs as keys and error (of test) as values
        path_result: path to save the plot as png
    """
    # Rearrange data to plot from dict
    lists_train = sorted(dico_error_train.items()) # sorted by key, return a list of tuples
    lists_test = sorted(dico_error_test.items())
    x_train, y_train = zip(*lists_train) # unpack a list of pairs into two tuples
    x_test, y_test = zip(*lists_test)
    # Plot data
    plt.plot(x_train, y_train, label="Train")
    plt.plot(x_test, y_test, label="Test")
    plt.title("Model Error per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(path_result, "error.png"))
    plt.close()

    return None

# Function: Plot error per IT, as a function of epoch
def plot_epoch_error_IT(dico_error: dict, path_result, flag_train=False):
    """
    Saves .png image: Error per IT, as a function of epoch
    Inputs:
        dico_error: a dict with epochs as keys and IT, error as values
        path_result: path to save the plot as png
    """
    dico_label = defaultdict(list)
    # First Loop: Rearrange data to plot it easily
    # dico: key="IT" / value="error"
    for key, value in dico_error.items():
        for k, v in value.items():
            dico_label[k].append(v)
    # Second Loop: Plot error for each IT
    for k, v in dico_label.items():
        plt.plot(list(range(len(dico_error))), v, label="IT "+ str(k))
    # Name of the file saved: train of test
    if(not flag_train):
        name_figure = "test"
    else:
        name_figure = "train"
    plt.title("Model Error per Time Interval, per Epoch - " + name_figure)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='upper right')
    print(os.path.join(path_result, "error_IT_" + name_figure + ".png"))
    plt.savefig(os.path.join(path_result, "error_IT_" + name_figure + ".png"))
    plt.close()

    return None

# Function: Custom Loss to remove -1 from computing
def my_custom_loss(output, target):
    """
    Function which compute Loss 
    Inputs: 
        output, target: prediction and labels
    Outputs: 
        loss
    """
    # Replace "-1" values by 1
    sub_target = torch.where(target != -1, target, 1.)
    sub_output = torch.where(target != -1, output.type(torch.float64), 100.)
    # Count number of elements with values "-1": these must not be considered
    num_removed = torch.sum( torch.where(target != -1, 0., 1.) )
    # Compute the loss (divide to get mean)
    the_loss = F.binary_cross_entropy_with_logits(sub_output, sub_target, reduction="sum")/(sub_output.numel()-num_removed)

    return the_loss
