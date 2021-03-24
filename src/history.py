'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import numpy as np
import os
import argparse
from time import time
from collections import defaultdict
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from .dataset import EHRHistoryDataset
from .utils import pretty_time, printc, create_session, save_json, get_label_threshold, mean_error
from .model_multiehr import MultiEHRModel
from .testing import test

def train_and_validate(train_loader, test_loader, device, config, path_result):
    model = MultiEHRModel(device, config)

    printc("\n----- STARTING TRAINING -----")

    losses = defaultdict(list)
    test_losses = []
    train_errors, test_errors = [], []
    n_samples = config.print_every_k_batch * config.batch_size
    for epoch in range(config.epochs):
        print("> EPOCH", epoch)
        model.train()
        k_batch_start_time = time()
        epoch_loss, k_batch_loss = 0, 0
        predictions, train_labels = [], []

        for i, (texts, dt, label) in enumerate(train_loader):
            if config.nrows and i*config.batch_size >= config.nrows:
                break
            loss, output = model.step(texts, dt, label)

            train_labels.append(label.item())
            predictions.append(output.item())

            epoch_loss += loss.item()
            k_batch_loss += loss.item()

            if (i+1) % config.print_every_k_batch == 0:
                average_loss = k_batch_loss / n_samples
                print('    [{}-{}]  -  Average loss: {:.4f}  -  Time elapsed: {}'.format(
                    i+1-config.print_every_k_batch, i+1, 
                    average_loss, 
                    pretty_time(time()-k_batch_start_time)
                ))
                losses[epoch].append(average_loss)
                k_batch_loss = 0
                k_batch_start_time = time()

        train_error = mean_error(train_labels, predictions, config.mean_time_survival)
        train_errors.append(train_error)
        printc(f'    Training mean error: {train_error:.2f} days - Global average loss: {epoch_loss/len(train_loader.dataset):.4f}\n', 'RESULTS')

        plt.plot(train_errors)
        plt.xlabel("Epoch")
        plt.ylabel("MAE (days)")
        plt.legend(["Train", "Validation"])
        plt.title("MAE Evolution")
        plt.savefig(os.path.join(config.path_result, "mae.png"))
        plt.close()
    
    printc("-----  Ended Training  -----\n")

    print("Saving losses...")
    save_json(path_result, "losses", { "train": losses, "validation": test_losses })
    plt.plot(np.linspace(0, config.epochs-1, sum([len(l) for l in losses.values()])),
             [ l for ll in losses.values() for l in ll ])
    plt.plot(test_losses)
    plt.legend(["Train loss", "Validation loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss evolution")
    plt.savefig(os.path.join(path_result, "loss.png"))
    plt.close()
    print("[DONE]")

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

    train_and_validate(train_loader, test_loader, device, config, config.path_result)

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