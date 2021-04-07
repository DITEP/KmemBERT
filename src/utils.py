'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import os
import torch
from datetime import datetime, date
import sys
import json
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

from .config import Config

def get_root():
    """
    Gets the absolute path to the root of the project
    """
    return os.sep.join(os.getcwd().split(os.sep)[0 : os.getcwd().split(os.sep).index("EHR_Transformers") + 1])

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
    return - mean_time_survival*np.log(1-label)

def shift_predictions(mus, mean_time_survival, shift):
    """
    Shift predictions in [0, 1] given a shift (transfer into survival time, 
    shift, and back into [0, 1])
    """
    time_survival = - mean_time_survival*torch.log(1-mus)
    time_survival -= shift
    return (1 - torch.exp(-time_survival/mean_time_survival)).clip(0, 1)

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
    
    print_args(args)

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

def mean_error(labels, predictions, mean_time_survival):
    """
    Computes the MAPE after transferring the labels and 
    predictions into survival time
    """
    return mean_absolute_percentage_error(
                label_to_time_survival(np.array(labels), mean_time_survival),
                label_to_time_survival(np.array(predictions), mean_time_survival))