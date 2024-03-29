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
import torch
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
    predictions, test_labels, stds, noigr = [], [], [], []
    test_start_time = time()

    total_loss = 0
    for _, (id, *data, labels) in enumerate(test_loader):
        noigr.append(id)
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
            if model.config.mode == 'density' or (model.config.mode == 'classif' and model.out_dim == 2):
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

    if test_losses is not None:
        test_losses.append(mean_loss)

    error = get_error(test_labels, predictions, config.mean_time_survival)
    printc(f"    {'Validation' if validation else 'Test'} | MAE: {int(error)} days - Global average loss: {mean_loss:.6f} - Time elapsed: {pretty_time(time()-test_start_time)}\n", 'RESULTS')

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
    save_json(path_result, "test", {"labels": test_labels, "predictions": predictions, "stds": stds, "noigr": noigr})

    predictions = np.array(predictions)
    test_labels = np.array(test_labels)

    if len(stds) > 0:
        n_points = 20
        resize_factor = 20
        gaussian_predictions = np.random.normal(predictions, np.array(stds)/resize_factor, size=(n_points, len(predictions))).flatten().clip(0, 1)
        associated_labels = np.tile(test_labels, n_points)
        sns.kdeplot(
            data={'Predictions': gaussian_predictions, 'Labels': associated_labels}, 
            y='Predictions', x='Labels', clip=((0, 1), (0, 1)),
            fill=True, thresh=0, levels=100, cmap="mako",
        )
        plt.title('Prediction distributions over labels')
        plt.savefig(os.path.join(path_result, "correlations_distributions.png"))
        plt.close()

        plt.scatter(test_labels, stds, s=0.1, alpha=0.5)
        plt.xlabel("Label")
        plt.ylabel("Standard Deviations")
        plt.xlim(0, 1)
        plt.ylim(0, max(stds))
        plt.title("Labels and corresponding standard deviations")
        plt.savefig(os.path.join(path_result, "stds.png"))
        plt.close()

    all_errors = get_error(test_labels, predictions, config.mean_time_survival, mean=False)

    plt.scatter(test_labels, all_errors, s=0.1, alpha=0.5)
    plt.xlabel("Labels")
    plt.ylabel("MAE")
    plt.xlim(0, 1)
    plt.ylim(0, max(all_errors))
    plt.title("MAE distribution")
    plt.savefig(os.path.join(path_result, "mae_distribution.png"))
    plt.close()

    plt.scatter(test_labels, predictions, s=0.1, alpha=0.5)
    plt.xlabel("Labels")
    plt.ylabel("Predictions")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Predictions / Labels correlation")
    plt.savefig(os.path.join(path_result, "correlations.png"))
    plt.close()

    errors_dict = defaultdict(list)
    for mae, label in zip(all_errors.tolist(), test_labels.tolist()):
        quantile = np.floor(label*10)
        errors_dict[quantile].append(mae)
    ape_per_quantile = sorted([(quantile, np.mean(l), np.std(l)) for quantile, l in errors_dict.items()])


    metrics = {}
    metrics["correlation"] = np.corrcoef(predictions, test_labels)[0,1]
    metrics["label_mae"] = np.mean(np.abs(predictions - test_labels))
    metrics["r2_score"] = r2_score(test_labels, predictions)
    
    for days in [30,90,180,270,360]:
        label = time_survival_to_label(days, config.mean_time_survival)
        bin_predictions = (predictions >= label).astype(int)
        bin_labels = (test_labels >= label).astype(int)
        
        days = f"{days} days"
        metrics[days] = {}
        metrics[days]['accuracy'] = accuracy_score(bin_labels, bin_predictions)
        metrics[days]['balanced_accuracy'] = balanced_accuracy_score(bin_labels, bin_predictions)
        metrics[days]['f1_score'] = f1_score(bin_labels, bin_predictions, average=None).tolist()
        
    try:
        # Error when only one class (in practice it happens only on the `ehr` sanity-check dataset)
        
        bin_labels = (test_labels >= 0.5).astype(int)
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
        printc("    Error while computing ROC curve", "WARNING")

    if not validation:
        print("Classification metrics:\n", metrics)

    save_json(path_result, 'results', 
        {'mae': error,
        'mean_loss': mean_loss,
        'metrics': metrics,
        'ape_per_quantile': ape_per_quantile})

    print(f"    (Ended {'validation' if validation else 'testing'})\n")
    return mean_loss



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