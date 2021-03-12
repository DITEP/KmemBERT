import numpy as np
import os
import argparse
from time import time
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

import torch
from torch.utils.data import DataLoader

from dataset import EHRDataset
from utils import pretty_time, printc, create_session, save_json, get_label_threshold, mean_error, label_to_time_survival
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
        
        if model.mode == 'classif':
            predictions += torch.softmax(outputs, dim=1).argmax(axis=1).tolist()
        elif model.mode == 'regression':
            predictions += outputs.flatten().tolist()
        else:
            mu, _ = outputs
            predictions += mu.tolist()
        
        test_labels += labels.tolist()
        total_loss += loss.item()

    if test_losses is not None:
        test_losses.append(total_loss/len(test_loader))

    error = mean_error(test_labels, predictions, config.mean_time_survival)
    printc(f"    {'Validation' if validation else 'Test'} mean error: {error:.2f} days -  Time elapsed: {pretty_time(time()-test_start_time)}\n", 'RESULTS')

    if validation:
        if error < model.best_error:
            model.best_error = error
            printc('    Best accuracy so far', 'SUCCESS')
            print('    Saving predictions...')
            save_json(path_result, "test", {"labels": test_labels, "predictions": predictions})
            print('    Saving model state...\n')
            state = {
                'model': model.state_dict(),
                'mean_error': error,
                'epoch': epoch,
                'tokenizer': model.tokenizer
            }
            torch.save(state, os.path.join(path_result, './checkpoint.pth'))
            model.early_stopping = 0
        else: 
            model.early_stopping += 1

    plt.scatter(predictions, test_labels, s=0.3, alpha=0.5)
    plt.xlabel("Predictions")
    plt.ylabel("Labels")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Predictions / Labels correlation")
    plt.savefig(os.path.join(path_result, "correlations.png"))
    plt.close()

    predictions = np.array(predictions)
    test_labels = np.array(test_labels)

    errors_dict = defaultdict(list)

    thresholds = [int(label_to_time_survival(0.1*quantile, config.mean_time_survival)) for quantile in range(1, 10)]
    quantiles = np.floor(test_labels*10).astype(int).tolist()

    for pred, label, quantile in zip(predictions.tolist(), test_labels.tolist(), quantiles):
        errors_dict[quantile].append(
            np.abs(label_to_time_survival(pred, config.mean_time_survival)
                   - label_to_time_survival(label, config.mean_time_survival)))

    std_mae_quantile = sorted([(quantile, np.std(l), np.mean(np.abs(l))) for quantile, l in errors_dict.items()])


    bin_predictions = (predictions >= config.label_threshold).astype(int)
    bin_labels = (test_labels >= config.label_threshold).astype(int)

    metrics = {}
    metrics['accuracy'] = accuracy_score(bin_labels, bin_predictions)
    metrics['balanced_accuracy'] = balanced_accuracy_score(bin_labels, bin_predictions)
    metrics['f1_score'] = f1_score(bin_labels, bin_predictions)
    metrics['confusion_matrix'] = confusion_matrix(bin_labels, bin_predictions).tolist()

    try:
        metrics['auc'] = roc_auc_score(bin_labels, predictions).tolist()

        fpr, tpr, thresholds = roc_curve(bin_labels, predictions)
        metrics['thresholds'] = thresholds.tolist()

        plt.plot(fpr, tpr)
        plt.xlabel("False Positive rate")
        plt.ylabel("True Positive rate")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("ROC curve")
        plt.savefig(os.path.join(path_result, "roc_curve.png"))
        plt.close()
    except:
        pass

    if not validation:
        print("Classification metrics:\n", metrics)

    save_json(path_result, 'results', 
        {'mean_error': error,
            'predictions': predictions.tolist(),
            'test_labels': test_labels.tolist(),
            'label_threshold': config.label_threshold,
            'bin_predictions': bin_predictions.tolist(),
            'bin_labels': bin_labels.tolist(),
            'metrics': metrics,
            'std_mae_quantile': std_mae_quantile})

    return error

def main(args):
    path_dataset, _, device, config = create_session(args)

    config.label_threshold = get_label_threshold(config, path_dataset)

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