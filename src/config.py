'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import json

class Config:
    """
    Stores useful information
    Many scripts uses a config instance. See utils.create_session for its initialization
    """

    voc_path = None
    model_name = "camembert-base"
    data_folder = None
    mode = "regression"
    print_every_k_batch = 8
    nrows = None
    batch_size = 64
    learning_rate = 1e-5
    epochs = 20
    freeze = False
    weight_decay = 0
    ratio_lr_embeddings = None
    drop_rate = None
    train_size = None
    path_result = None
    resume = None
    patience = None
    days_threshold = 90
    max_ehrs = None

    def __init__(self, args):
        for attr in dir(self):
            if not attr.startswith('__') and hasattr(args, attr):
                setattr(self, attr, getattr(args, attr))

    def __repr__(self):
        return json.dumps(vars(self), sort_keys=True, indent=4)