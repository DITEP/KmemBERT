'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import argparse
import torch
from torch.utils.data import DataLoader

from .dataset import EHRDataset
from .utils import create_session, get_label_threshold
from .models import HealthBERT

def analyse(model, text):
    return 0

def main(args):
    path_dataset, _, device, config = create_session(args)

    config.label_threshold = get_label_threshold(config, path_dataset)

    _, validation_dataset = EHRDataset.get_train_validation(path_dataset, config)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=True)

    model = HealthBERT(device, config)
    model.train(mode=False)

    analyse_dict = {}

    for i, (*data, labels) in enumerate(validation_loader):
        _, outputs = model.step(*data, labels)
        errors = torch.abs(outputs[0] - labels)

        for text, error in zip(data[0], errors):
            analyse_dict[error] = analyse(model, text)

    for _, value in sorted(analyse_dict.items())[:10]:
        print(value)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
        help="data folder name")
    parser.add_argument("-m", "--mode", type=str, default="regression", choices=['regression', 'density'],
        help="name of the task")
    parser.add_argument("-b", "--batch_size", type=int, default=8, 
        help="dataset batch size")
    parser.add_argument("-nr", "--nrows", type=int, default=None, 
        help="maximum number of samples for training and validation")
    parser.add_argument("-dt", "--days_threshold", type=int, default=365, 
        help="days threshold to convert into classification task")
    parser.add_argument("-r", "--resume", type=str, default=None, 
        help="result folder in which the saved checkpoint will be reused")
    parser.add_argument("-pd", "--path_dataset", type=str, default=None, 
        help="custom path to the dataset, in case the data folder wouldn't be in KmemBERT/data/")

    main(parser.parse_args())