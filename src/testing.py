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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from .dataset import EHRDataset
from .utils import pretty_time, printc, create_session, save_json, get_label_threshold, mean_error, label_to_time_survival
from .models.health_bert import HealthBERT

def test(model, test_loader, config, path_result, epoch=-1, test_losses=None, validation=False):
    """
    Tests a model on a test_loader and compute its accuracy
    """
    
    model.eval()
    predictions, test_labels, stds = [], [], []
    test_start_time = time()

    total_loss = 0
    for _, (*data, labels) in enumerate(test_loader):
        loss, outputs = model.step(*data, labels)
        
        if model.mode == 'classif':
            predictions += torch.softmax(outputs, dim=1).argmax(axis=1).tolist()
        elif model.mode == 'regression':
            predictions += outputs.flatten().tolist()
        elif model.mode == 'density':
            mus, log_vars = outputs
            predictions += mus.tolist()
            stds += torch.exp(log_vars/2).tolist()
        elif model.mode == 'multi':
            if model.config.mode in ['density', 'classif']:
                mu, log_var = outputs
                predictions.append(mu.item())
                stds.append(torch.exp(log_var/2).item())
            else:
                predictions.append(outputs.item())
        else:
            raise ValueError(f'Mode {model.mode} unknown')
        
        test_labels += labels.tolist()
        total_loss += loss.item()
    
    mean_loss = total_loss/(config.batch_size*len(test_loader))
    print(f"    {'Validation' if validation else 'Test'} loss: {mean_loss:.2f}")

    if test_losses is not None:
        test_losses.append(mean_loss)

    error = mean_error(test_labels, predictions, config.mean_time_survival)
    printc(f"    {'Validation' if validation else 'Test'} | mean error: {error:.2f} days - Global average loss: {mean_loss:.4f} - Time elapsed: {pretty_time(time()-test_start_time)}\n", 'RESULTS')

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

    print('    Saving predictions...\n')
    save_json(path_result, "test", {"labels": test_labels, "predictions": predictions, "stds": stds})

    if len(stds) > 0:
        n_points = 20
        resize_factor = 20
        gaussian_predictions = np.random.normal(predictions, np.array(stds)/resize_factor, size=(n_points, len(predictions))).flatten().clip(0, 1)
        associated_labels = np.tile(test_labels, n_points)
        sns.kdeplot(
            data={'Predictions': gaussian_predictions, 'Labels': associated_labels}, 
            x='Predictions', y='Labels', clip=((0, 1), (0, 1)),
            fill=True, thresh=0, levels=100, cmap="mako",
        )
        plt.title('Prediction distributions over labels')
        plt.savefig(os.path.join(path_result, "correlations_distributions.png"))
        plt.close()

        plt.scatter(predictions, stds, s=0.1, alpha=0.5)
        plt.xlabel("Predictions")
        plt.ylabel("Standard Deviations")
        plt.xlim(0, 1)
        plt.ylim(0, max(stds))
        plt.title("Predictions and corresponding standard deviations")
        plt.savefig(os.path.join(path_result, "stds.png"))
        plt.close() 

    plt.scatter(predictions, test_labels, s=0.1, alpha=0.5)
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
    metrics['f1_score'] = f1_score(bin_labels, bin_predictions, average=None).tolist()
    metrics['confusion_matrix'] = confusion_matrix(bin_labels, bin_predictions).tolist()
    metrics['correlation'] = float(np.corrcoef(predictions, test_labels)[0,1])

    try:
        metrics['auc'] = roc_auc_score(bin_labels, predictions).tolist()

        fpr, tpr, _ = roc_curve(bin_labels, predictions)

        plt.plot(fpr, tpr)
        plt.plot([0,1], [0,1], 'r--')
        plt.xlabel("False Positive rate")
        plt.ylabel("True Positive rate")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("ROC curve")
        plt.savefig(os.path.join(path_result, "roc_curve.png"))
        plt.close()
    except:
        print("Error while computing ROC curve...")

    if not validation:
        print("Classification metrics:\n", metrics)

    save_json(path_result, 'results', 
        {'mean_error': error,
        'metrics': metrics,
        'mean_loss': mean_loss,
        'predictions': predictions.tolist(),
        'test_labels': test_labels.tolist(),
        'label_threshold': config.label_threshold,
        'bin_predictions': bin_predictions.tolist(),
        'bin_labels': bin_labels.tolist(),
        'std_mae_quantile': std_mae_quantile})

    return mean_loss

def main(args):
    path_dataset, _, device, config = create_session(args)

    config.label_threshold = get_label_threshold(config, path_dataset)

    dataset = EHRDataset(path_dataset, config, train=False)
    loader = DataLoader(dataset, batch_size=config.batch_size)

    model = HealthBERT(device, config)

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