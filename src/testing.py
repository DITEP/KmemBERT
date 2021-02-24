import numpy as np
import os
import argparse
from time import time
import matplotlib.pyplot as plt
import json

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix

import torch
from torch.utils.data import DataLoader

from dataset import EHRDataset
from utils import pretty_time, printc, create_session, save_json, label_to_time_survival, time_survival_to_label
from health_bert import HealthBERT

def test(model, test_loader, config, path_result, epoch=-1, test_losses=None, validation=False):
    """
    Tests a model on a test_loader and compute its accuracy
    """
    
    model.eval()
    predictions, test_labels = [], []
    test_start_time = time()

    total_loss = 0
    for _, (texts, labels) in enumerate(test_loader):
        loss, outputs = model.step(texts, labels)
        
        if model.classify:
            predictions += torch.softmax(outputs, dim=1).argmax(axis=1).tolist()
        else:
            predictions += outputs.flatten().tolist()
        
        test_labels += labels.tolist()
        total_loss += loss.item()

    if test_losses is not None:
        test_losses.append(total_loss/len(test_loader))

    mean_error = np.abs(label_to_time_survival(np.array(test_labels), config.mean_time_survival) - 
                  label_to_time_survival(np.array(predictions), config.mean_time_survival)).mean()
    printc(f"    {'Validation' if validation else 'Test'} mean error: {mean_error:.3f} days -  Time elapsed: {pretty_time(time()-test_start_time)}\n", 'RESULTS')

    if validation:
        if mean_error < model.best_error:
            model.best_error = mean_error
            printc('    Best accuracy so far', 'SUCCESS')
            print('    Saving predictions...')
            save_json(path_result, "test", {"labels": test_labels, "predictions": predictions})
            print('    Saving model state...\n')
            state = {
                'model': model.state_dict(),
                'mean_error': mean_error,
                'epoch': epoch,
                'tokenizer': model.tokenizer
            }
            torch.save(state, os.path.join(path_result, './checkpoint.pth'))
            model.early_stopping = 0
        else: 
            model.early_stopping += 1
    else:
        plt.scatter(predictions, test_labels)
        plt.xlabel("Predictions")
        plt.ylabel("Labels")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("Predictions / Labels correlation")
        plt.savefig(os.path.join(path_result, "correlations.png"))

        predictions = np.array(predictions)
        test_labels = np.array(test_labels)

        bin_predictions = (predictions >= config.label_threshold).astype(int)
        bin_labels = (test_labels >= config.label_threshold).astype(int)

        metrics = {}
        metrics['accuracy'] = accuracy_score(bin_labels, bin_predictions)
        metrics['balanced_accuracy'] = balanced_accuracy_score(bin_labels, bin_predictions)
        metrics['f1_score'] = f1_score(bin_labels, bin_predictions)
        metrics['auc'] = roc_auc_score(bin_labels, predictions)
        metrics['confusion_matrix'] = confusion_matrix(bin_labels, bin_predictions).tolist()
        print("Classification metrics:\n", metrics)

        save_json(path_result, 'results', 
            {'mean_error': mean_error,
             'predictions': predictions.tolist(),
             'test_labels': test_labels.tolist(),
             'label_threshold': config.label_threshold,
             'bin_predictions': bin_predictions.tolist(),
             'bin_labels': bin_labels.tolist(),
             'metrics': metrics})

    return mean_error

def main(args):
    path_dataset, _, device, config = create_session(args)

    config_path = os.path.join(path_dataset, "config.json")
    assert os.path.isfile(config_path), 'Config file not existing, please train a model first'
    with open(config_path) as json_file:
        mean_time_survival = json.load(json_file)["mean_time_survival"]

    config.label_threshold = time_survival_to_label(args.days_threshold, mean_time_survival)

    dataset = EHRDataset(path_dataset, config, train=False)
    loader = DataLoader(dataset, batch_size=config.batch_size)

    model = HealthBERT(device, config)
    printc(f"Resuming with model at {config.resume}", "INFO")
    path_checkpoint = os.path.join(os.path.dirname(config.path_result), config.resume, 'checkpoint.pth')
    assert os.path.isfile(path_checkpoint), 'Error: no checkpoint found!'
    checkpoint = torch.load(path_checkpoint, map_location=model.device)
    model.load_state_dict(checkpoint['model'])
    model.tokenizer = checkpoint['tokenizer']

    test(model, loader, config, config.path_result)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
        help="data folder name")
    parser.add_argument("-r", "--resume", type=str, required=True, 
        help="result folder in with the saved checkpoint will be reused")
    parser.add_argument("-dt", "--days_threshold", type=int, default=90, 
        help="days threshold to convert into classification task")
    parser.add_argument("-b", "--batch_size", type=int, default=8, 
        help="dataset batch size")
    parser.add_argument("-nr", "--nrows", type=int, default=None, 
        help="maximum number of samples for training and testing")

    main(parser.parse_args())