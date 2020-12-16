import os
import platform

def get_root():
    return os.sep.join(os.getcwd().split(os.sep)[0 : os.getcwd().split(os.sep).index("French-Transformers") + 1])

def pretty_time(t):
    # Tranforms time t in seconds into a pretty string
    return f"{int(t//60)}m{int(t%60)}s"