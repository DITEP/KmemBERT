'''
    Author: CentraleSupelec, Théo Di Piazza (CLB)
    Year: 2022
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
import torch
import math
from torch.utils.data import DataLoader
import json

from .dataset import EHRDataset, PredictionsDataset
from .utils import pretty_time, printc, create_session, save_json, get_label_threshold, get_error, time_survival_to_label, collate_fn, collate_fn_with_id
from .models import HealthBERT, TransformerAggregator, Conflation, SanityCheck

def test(model, test_loader, config, path_result, epoch=-1, test_losses=None, validation=False):
    """
    Tests a model on a test_loader and compute its accuracy
    """
    
    model.eval()
    predictions, test_labels = [], []
    test_epoch_loss = []
    indice_followup = []
    test_start_time = time()

    for i, data in enumerate(test_loader):

        # Get features with good format
        embedding = data[0][0] # Embedding
        dt = torch.tensor(data[0][1], dtype=torch.float32) # dt (difference time)
        labels = torch.tensor(data[0][2], dtype=torch.float32) # Label as tensor
        myRowIndice = int(data[1].item()) # Indice of the raw
        flagRecent = int(data[2].item()) # FlagRecent

        # Use this row to test if it's the most recent one
        if(flagRecent):
            # Get indice of current raw to follow-up
            indice_followup.append(myRowIndice)

            # Mise au bon format pour le step
            data_to_step = ([embedding], dt, labels)

            # Step function: get loss and outputs
            loss, outputs = model.step(*data_to_step)

            # Save loss, label and prediction from the model
            mu, _ = outputs
            predictions.append(mu.item())
            test_labels.append(labels.item())
            test_epoch_loss.append(loss.item())

    # Compute MEAN and SUM of loss of this EPOCH
    test_mean_loss_epoch = np.sum(test_epoch_loss)/len(test_epoch_loss)
    test_sum_loss_epoch = np.sum(test_epoch_loss)

    printc(f'    Testing | Epoch: {epoch} - Mean Loss: {test_mean_loss_epoch:.6f} - Time elapsed: {pretty_time(time()-test_start_time)}\n', 'RESULTS')

    if validation:
        if test_mean_loss_epoch < model.best_loss:
            model.best_loss = test_mean_loss_epoch
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
            return test_mean_loss_epoch, test_sum_loss_epoch

    # Save predictions and test_labels as json files
    print('    Saving predictions...   ')
    pred_and_labels = {}
    pred_and_labels['predictions'] = predictions
    pred_and_labels['test_labels'] = test_labels
    pred_and_labels['indiceFollowUp'] = indice_followup
    save_json(path_result, 'pred_and_labels', {'test_results': pred_and_labels})
    printc('   Predictions and Labels saved.   ', 'SUCCESS')

    print(f"    (Ended {'validation' if validation else 'testing'})\n")

    return test_mean_loss_epoch, test_sum_loss_epoch



def main(args):
    path_dataset, _, device, config = create_session(args)

    config.label_threshold = get_label_threshold(config, path_dataset)

    with open(os.path.join('results', config.resume, 'args.json')) as json_file:
        training_args = json.load(json_file)

    if 'mode' in training_args.keys():
        model = HealthBERT(device, config)
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
        config.resume = training_args['resume']
        dataset = PredictionsDataset(path_dataset, config, train=False, device=device, output_hidden_states=(aggregator == 'transformer'), return_id=True)
        loader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=collate_fn_with_id)
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