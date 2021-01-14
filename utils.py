import os
import torch

def get_root():
    return os.sep.join(os.getcwd().split(os.sep)[0 : os.getcwd().split(os.sep).index("French-Transformers") + 1])

def pretty_time(t):
    """Tranforms time t in seconds into a pretty string"""
    return f"{int(t//60)}m{int(t%60)}s"

def time_survival_to_label(ts, mean_time_survival):
    """Transforms times of survival into uniform labels in ]0,1["""
    return 1 - torch.exp(-ts/mean_time_survival)

def label_to_time_survival(label, mean_time_survival):
    """Transforms labels in ]0,1[ into times of survival"""
    return - mean_time_survival*torch.log(1-label)