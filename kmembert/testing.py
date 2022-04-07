'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import numpy as np
import os
import argparse
from time import time
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve, r2_score
import seaborn as sns
import itertools
import torch
from torch.utils.data import DataLoader
import json

from .dataset import EHRDataset
from .utils import pretty_time, printc, create_session, save_json, get_label_threshold, get_error, get_error_IT, time_survival_to_label, collate_fn, collate_fn_with_id
from .models import CamembertRegressor, TransformerAggregator, Conflation, SanityCheck

def test(model, test_loader, config, path_result, epoch=-1, test_losses=None, validation=False):
    """
    Tests a model on a test_loader and compute its accuracy
    """
    
    model.eval()
    predictions, test_labels, stds, noigr = [], [], [], []
    test_start_time = time()
    total_loss = 0
    test_labels_IT, test_outputs_IT = [], [] # Error per IT

    for id, (*data, labels) in enumerate(test_loader):
        noigr.append(id)

        # Make prediction: Get Loss and Outputs
        loss, outputs = model.step(*data, labels)

        # Stock predictions and labels
        labels = labels.tolist(); outputs = outputs.tolist();

        # To get error per time interval
        test_labels_IT += labels
        test_outputs_IT += outputs

        # To get overall error
        test_labels += list(itertools.chain(*labels))
        predictions += list(itertools.chain(*outputs))
        
        # Compute loss
        total_loss += loss.item()
    
    # Get mean loss and error of the prediction
    mean_loss = total_loss/len(test_loader)

    # Get overall and per IT error
    error_test = get_error(np.array(test_labels), np.array(predictions))
    error_IT_test = get_error_IT(np.array(test_labels_IT), np.array(test_outputs_IT))
    

    if test_losses is not None:
        test_losses.append(mean_loss)

    printc(f"    {'Validation' if validation else 'Test'} | Error: {error_test} - Global average loss: {mean_loss:.6f} - Time elapsed: {pretty_time(time()-test_start_time)}\n", 'RESULTS')

    if validation:
        if mean_loss < model.best_loss:
            model.best_loss = mean_loss
            printc('    Best loss so far', 'SUCCESS')
            print('    Saving model state...')
            state = {
                'model': model.state_dict(),
                'optimizer': model.optimizer.state_dict(),
                'scheduler': model.scheduler.state_dict(),
                'best_loss': model.best_loss,
                'epoch': epoch,
                'tokenizer': model.tokenizer if hasattr(model, 'tokenizer') else None
            }
            torch.save(state, os.path.join(path_result, './checkpoint.pth'))
            model.early_stopping = 0
        else: 
            model.early_stopping += 1
            return mean_loss

    print('    Saving predictions...')
    save_json(path_result, "test", {"labels": test_labels, "predictions": predictions})


    return mean_loss, error_test, error_IT_test



def main(args):
    path_dataset, _, device, config = create_session(args)

    config.label_threshold = get_label_threshold(config, path_dataset)

    with open(os.path.join('results', config.resume, 'args.json')) as json_file:
        training_args = json.load(json_file)

    if 'mode' in training_args.keys():
        model = CamembertRegressor(device, config)
    else:
        aggregator = training_args['aggregator']
        config.max_ehrs = training_args['max_ehrs']

        if aggregator == 'transformer':
            model = TransformerAggregator(device, config, training_args['nhead'], training_args['num_layers'], training_args['out_dim'], training_args['time_dim'])
            model.initialize_scheduler()
            model.resume(config)

        elif aggregator == 'conflation':
            model = Conflation(device, config)

        elif aggregator == 'sanity_check':
            model = SanityCheck(device, config)


    if model.mode == 'multi':
        None
    else:
        dataset = EHRDataset(path_dataset, config, train=False, return_id=True)
        loader = DataLoader(dataset, batch_size=config.batch_size)

    test(model, loader, config, config.path_result)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
        help="data folder name")
    parser.add_argument("-r", "--resume", type=str, required=True, 
        help="result folder in with the saved checkpoint will be reused")
    parser.add_argument("-dt", "--days_threshold", type=int, default=365, 
        help="days threshold to convert into classification task")
    parser.add_argument("-b", "--batch_size", type=int, default=8, 
        help="dataset batch size")
    parser.add_argument("-nr", "--nrows", type=int, default=None, 
        help="maximum number of samples for training and testing")

    main(parser.parse_args())