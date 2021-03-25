'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import argparse
import torch
from torch.utils.data import DataLoader

from .dataset import EHRHistoryDataset
from .utils import create_session, get_label_threshold
from .models.multi_ehr import MultiEHR
from .training import train_and_validate

def collate_fn(batch):
    sequences, times, labels = zip(*batch)
    return sequences, torch.tensor(times).type(torch.float32), torch.tensor(labels).type(torch.float32)

def main(args):
    path_dataset, _, device, config = create_session(args)

    config.label_threshold = get_label_threshold(config, path_dataset)

    dataset = EHRHistoryDataset(path_dataset, config)
    train_size = int(config.train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    model = MultiEHR(device, config)
    train_and_validate(model, train_loader, test_loader, device, config, config.path_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
        help="data folder name")
    parser.add_argument("-e", "--epochs", type=int, default=2, 
        help="number of epochs")
    parser.add_argument("-t", "--train_size", type=float, default=0.8, 
        help="dataset train size")
    parser.add_argument("-nr", "--nrows", type=int, default=None, 
        help="maximum number of samples for training and testing")
    parser.add_argument("-k", "--print_every_k_batch", type=int, default=1, 
        help="maximum number of samples for training and testing")
    parser.add_argument("-f", "--freeze", type=bool, default=False, const=True, nargs="?",
        help="whether or not to freeze the Bert part")
    parser.add_argument("-dt", "--days_threshold", type=int, default=90, 
        help="days threshold to convert into classification task")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, 
        help="dataset train size")
    parser.add_argument("-wg", "--weight_decay", type=float, default=0, 
        help="the weight decay for L2 regularization")
    parser.add_argument("-r", "--resume", type=str, required=True, 
        help="result folder in which the saved checkpoint will be reused")
    parser.add_argument("-p", "--patience", type=int, default=4, 
        help="Number of decreasing accuracy epochs to stop the training")

    main(parser.parse_args())