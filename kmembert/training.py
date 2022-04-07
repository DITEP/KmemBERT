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
import itertools

import torch
from torch.utils.data import DataLoader

from .dataset import EHRDataset
from .utils import plot_epoch_error_IT, pretty_time, printc, create_session, save_json, get_label_threshold, get_error, get_error_IT, plot_epoch_loss, plot_epoch_error, plot_epoch_error_IT
from .models import CamembertRegressor
from .testing import test

def train_and_validate(model, train_loader, validation_loader, device, config, path_result):
    """
    Creates a camembert model and retrain it, with eventually a larger vocabulary.

    Inputs: please refer bellow, to the argparse arguments.
    """
    printc("\n----- STARTING TRAINING -----")

    losses, errors, errors_IT = {}, {}, {}# Stock mean loss, error, error per time interval; per epoch - for training
    errors_test, errors_test_IT = {}, {} # Stock error, error per time interval; per epoch - for testing
    validation_losses = []
    n_samples = config.print_every_k_batch * config.batch_size
    model.initialize_scheduler(config.epochs, train_loader)
    for epoch in range(config.epochs):
        print("> EPOCH", epoch)
        model.train()
        epoch_loss, k_batch_loss = 0, 0
        epoch_start_time, k_batch_start_time = time(), time()
        predictions, train_labels = [], []
        train_labels_IT, train_outputs_IT = [], [] # Error per IT

        for i, (*data, labels) in enumerate(train_loader):
            # Labels to tensor
            #labels = torch.tensor(labels, dtype=torch.float)

            if config.nrows and i*config.batch_size >= config.nrows:
                break

            # Make prediction: Get Loss and Outputs
            loss, outputs = model.step(*data, labels)

            # Stock predictions and labels
            labels = labels.tolist(); outputs = outputs.tolist();

            # To get error per time interval
            train_labels_IT += labels
            train_outputs_IT += outputs
            # To get overall error
            train_labels += list(itertools.chain(*labels))
            predictions += list(itertools.chain(*outputs))

            # Compute loss
            epoch_loss += loss.item()
            k_batch_loss = loss.item()
            
            print(f'    [{i+1-config.print_every_k_batch}-{i+1}]  -  Average loss: {k_batch_loss:.6f}  -  Time elapsed: {pretty_time(time()-k_batch_start_time)}')
            k_batch_start_time = time()

        # Get loss and error of the epoch
        mean_loss = epoch_loss/len(train_loader)
        losses.update( {epoch: mean_loss} )
        # Get overall and per IT error
        error = get_error(np.array(train_labels), np.array(predictions))
        error_IT = get_error_IT(np.array(train_labels_IT), np.array(train_outputs_IT))
        # Stock the error
        errors.update( {epoch: error} )
        errors_IT.update( {epoch: error_IT} )
        printc(f'    Training   | Error: {error} - Global average loss: {mean_loss:.6f} - Time elapsed: {pretty_time(time()-epoch_start_time)}\n', 'RESULTS')

        # Test the model at each batch
        _, error_test, error_test_IT = test(model, validation_loader, config, config.path_result, epoch=epoch, test_losses=validation_losses, validation=True)

        # Stock the test error 
        errors_test.update( {epoch: error_test} )
        errors_test_IT.update( {epoch: error_test_IT})

        model.scheduler.step()
        if (config.patience is not None) and (model.early_stopping >= config.patience):
            printc(f'Breaking training after patience {config.patience} reached', 'INFO')
            break

    print("errors_test at the end:", errors_test)
    print("errors test per IT at the end:", errors_test_IT)

    # Saving results
    # Save plot: loss as a function of epoch (train and test)
    plot_epoch_loss(losses, validation_losses, path_result)

    # Save plot: error as a function of epoch (test)
    plot_epoch_error(errors, errors_test, path_result)
    
    # Save plot: error per ITas a function of epoch (test)
    plot_epoch_error_IT(errors_IT, path_result, flag_train=False) # For train dataset
    plot_epoch_error_IT(errors_test_IT, path_result, flag_train=True) # For test dataset

    printc("-----  Ended Training  -----\n")

    print("[DONE]")

    return model.best_loss

def main(args):
    print("args :", type(args))
    path_dataset, _, device, config = create_session(args)

    assert not (args.freeze and args.voc_file), "Don't use freeze argument while adding vocabulary. It would not be learned"

    config.label_threshold = get_label_threshold(config, path_dataset)

    train_dataset, validation_dataset = EHRDataset.get_train_validation(path_dataset, config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=2, shuffle=True)

    model = CamembertRegressor(device, config)
    train_and_validate(model, train_loader, validation_loader, device, config, config.path_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
        help="data folder name")
    parser.add_argument("-m", "--mode", type=str, default="regression", choices=['regression', 'density', 'classif'],
        help="name of the task")
    parser.add_argument("-b", "--batch_size", type=int, default=8, 
        help="dataset batch size")
    parser.add_argument("-e", "--epochs", type=int, default=2, 
        help="number of epochs")
    parser.add_argument("-drop", "--drop_rate", type=float, default=0.1, 
        help="dropout ratio. By default, None uses p=0.1")
    parser.add_argument("-nr", "--nrows", type=int, default=None, 
        help="maximum number of samples for training and validation")
    parser.add_argument("-k", "--print_every_k_batch", type=int, default=1, 
        help="prints training loss every k batch")
    parser.add_argument("-f", "--freeze", type=bool, default=False, const=True, nargs="?",
        help="whether or not to freeze the Bert part")
    parser.add_argument("-dt", "--days_threshold", type=int, default=365, 
        help="days threshold to convert into classification task")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, 
        help="model learning rate")
    parser.add_argument("-r_lr", "--ratio_lr_embeddings", type=float, default=1, 
        help="the ratio applied to lr for embeddings layer")
    parser.add_argument("-wg", "--weight_decay", type=float, default=0, 
        help="the weight decay for L2 regularization")
    parser.add_argument("-v", "--voc_file", type=str, default=None, 
        help="voc file containing camembert added vocabulary")
    parser.add_argument("-r", "--resume", type=str, default=None, 
        help="result folder in which the saved checkpoint will be reused")
    parser.add_argument("-p", "--patience", type=int, default=4, 
        help="number of decreasing accuracy epochs to stop the training")
    parser.add_argument("-nl", "--num_label", type=int, default=2, 
        help="number of label to predict")

    main(parser.parse_args())