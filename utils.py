import os
import platform
import torch

def get_root():
    return os.sep.join(os.getcwd().split(os.sep)[0 : os.getcwd().split(os.sep).index("French-Transformers") + 1])

def pretty_time(t):
    # Tranforms time t in seconds into a pretty string
    return f"{int(t//60)}m{int(t%60)}s"

def time_survival_to_label(ts, mean_time_survival):
    return 1 - torch.exp(-ts/mean_time_survival)

def label_to_time_survival(label, mean_time_survival):
    return - mean_time_survival*torch.log(1-label)