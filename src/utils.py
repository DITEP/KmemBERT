import os
import torch
from datetime import datetime
import sys
import json

def get_root():
    return os.sep.join(os.getcwd().split(os.sep)[0 : os.getcwd().split(os.sep).index("EHR_Transformers") + 1])

def pretty_time(t):
    """Tranforms time t in seconds into a pretty string"""
    return f"{int(t//60)}m{int(t%60)}s"

def time_survival_to_label(ts, mean_time_survival):
    """Transforms times of survival into uniform labels in ]0,1["""
    return 1 - torch.exp(-ts/mean_time_survival)

def label_to_time_survival(label, mean_time_survival):
    """Transforms labels in ]0,1[ into times of survival"""
    return - mean_time_survival*torch.log(1-label)

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
    print(f"{bcolors[color]}{log}{bcolors['ENDC']}")

def now():
    return datetime.now().strftime('%y-%m-%d_%Hh%Mm%Ss')

def create_session(args):
    torch.manual_seed(0)

    print(f"> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    printc(f"> DEVICE:  {device}", "INFO")

    path_root = get_root()
    printc(f"> ROOT:    {path_root}", "INFO")

    path_dataset = os.path.join(path_root, "data", args.dataset)

    main_file = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    session_id = f"{main_file}_{now()}"
    path_result = os.path.join(path_root, "results", session_id)
    os.mkdir(path_result)
    printc(f"> SESSION: {path_result}", "INFO")

    with open(os.path.join(path_result, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)

    return path_dataset, path_result, device