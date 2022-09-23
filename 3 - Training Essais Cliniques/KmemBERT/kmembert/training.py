'''
    Author: CentraleSupelec, Théo Di Piazza (Centre Léon Bérard)
    Year: 2021, 2022
    Python Version: >= 3.7
'''

import numpy as np
import os
import argparse
import math
from time import time
from collections import defaultdict
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from .dataset import EHRDataset
from .utils import pretty_time, printc, printch, create_session, save_json, get_label_threshold, get_error
from .models import HealthBERT
from .testing import test

def train_and_validate(model, train_loader, validation_loader, device, config, path_result, train_only=False):
    """
    Creates a camembert model and retrain it, with eventually a larger vocabulary.

    Inputs: please refer bellow, to the argparse arguments.
    """
    printc("\n----- STARTING TRAINING -----")

    losses = defaultdict(list)
    validation_losses = []
    loss_train_mean, loss_test_mean = {}, {}
    loss_train_sum, loss_test_sum = {}, {}
    n_samples = config.print_every_k_batch * config.batch_size
    model.initialize_scheduler(config.epochs, train_loader)
    for epoch in range(config.epochs):
        printch(f"> EPOCH {epoch}/{config.epochs-1}", "white", "blue")
        model.train()
        train_epoch_loss = []
        epoch_start_time = time()
        predictions, train_labels, indice_followup = [], [], []

        for i, data in enumerate(train_loader):
            if config.nrows and i*config.batch_size >= config.nrows:
                break

            # Get features with good format
            embedding = data[0][0] # Embedding
            dt = torch.tensor(data[0][1], dtype=torch.float32) # dt (difference time)
            labels = torch.tensor(data[0][2], dtype=torch.float32) # Label as tensor
            myRowIndice = int(data[1].item()) # Indice of the raw
            flagRecent = int(data[2].item()) # FlagRecent

            # Use this row to train if it's the most recent
            if(flagRecent):
                # Mise au bon format pour le step
                data_to_step = ([embedding], dt, labels)
                
                # Step function: get loss and outputs
                loss, outputs = model.step(*data_to_step)

                # Save loss, label and prediction from the model
                mu, _ = outputs
                predictions.append(mu.item())
                train_labels.append(labels.item())
                train_epoch_loss.append(loss.item())

        # Compute MEAN and SUM of loss of this EPOCH
        train_mean_loss_epoch = np.sum(train_epoch_loss)/len(train_epoch_loss)
        train_sum_loss_epoch = np.sum(train_epoch_loss)

        printc(f'    Training | Epoch: {epoch} - Mean Loss: {train_mean_loss_epoch:.6f} - Time elapsed: {pretty_time(time()-epoch_start_time)}', 'RESULTS')

        # Test the model
        test_mean_loss_epoch, test_sum_loss_epoch = test(model, validation_loader, config, config.path_result, epoch=epoch, test_losses=validation_losses, validation=True)
        # Save Loss Results for Train & Test for the current epoch
        loss_test_mean[epoch] = test_mean_loss_epoch # Loss of test - Mean
        loss_train_mean[epoch] = train_mean_loss_epoch # Loss of TRAIN - Mean
        loss_test_sum[epoch] = test_sum_loss_epoch # Loss of test - Sum
        loss_train_sum[epoch] = train_sum_loss_epoch # Loss of TRAIN - Sum

        # Update scheduler of the mode (learning rate)
        model.scheduler.step()
        '''
        if (config.patience is not None) and (model.early_stopping >= config.patience):
            printc(f'Breaking training after patience {config.patience} reached', 'INFO')
            break
        '''
    
    printc("-----  Ended Training  -----\n")

    # Save loss
    print("   Saving losses...   ")
    save_json(path_result, 'losses_mean', {'train': loss_train_mean, 'test': loss_test_mean})
    save_json(path_result, 'losses_sum', {'train': loss_train_sum, 'test': loss_test_sum})

    # Save plot
    plt.plot(list(range(epoch+1)), list(loss_train_mean.values()), label='Train')
    plt.plot(list(range(epoch+1)), list(loss_test_mean.values()), label='Test')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Loss")
    plt.title("Mean Loss for each epoch")
    plt.legend()
    plt.savefig(os.path.join(path_result, "mean_loss_plot.png"), bbox_inches="tight")
    plt.close()

    printc("   Losses saved..."   , "SUCCESS")
    

    return model.best_loss

def main(args):
    path_dataset, _, device, config = create_session(args)

    assert not (args.freeze and args.voc_file), "Don't use freeze argument while adding vocabulary. It would not be learned"

    config.label_threshold = get_label_threshold(config, path_dataset)

    train_dataset, validation_dataset = EHRDataset.get_train_validation(path_dataset, config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

    model = HealthBERT(device, config)
    train_and_validate(model, train_loader, validation_loader, device, config, config.path_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
        help="data folder name")
    parser.add_argument("-m", "--mode", type=str, default="regression", choices=['regression', 'density'],
        help="name of the task")
    parser.add_argument("-b", "--batch_size", type=int, default=8, 
        help="dataset batch size")
    parser.add_argument("-e", "--epochs", type=int, default=2, 
        help="number of epochs")
    parser.add_argument("-drop", "--drop_rate", type=float, default=None, 
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

    main(parser.parse_args())