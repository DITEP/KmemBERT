'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import argparse

import optuna
import torch
from torch.utils.data import DataLoader

from .utils import create_session, get_label_threshold
from .training import train_and_validate
from .dataset import EHRDataset


def main(args):
    path_dataset, path_result, device, config = create_session(args)
    config.label_threshold = get_label_threshold(config, path_dataset)

    dataset = EHRDataset(path_dataset, config)
    train_size = int(config.train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    def objective(trial):
        config.batch_size = 8 # trial.suggest_categorical('batch_size', [32, 64, 128])
        config.learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-4)
        config.freeze = False # trial.suggest_categorical('freeze', [False, True])
        config.weight_decay = 0 # trial.suggest_categorical('weight_decay', [0,1e-2,1e-1])
        config.drop_rate = 0 # trial.suggest_categorical('drop_rate', [0., 0.1, 0.2])

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

        print(f"\nTrial config: {config}")
        return train_and_validate(train_loader, test_loader, device, config, path_result, train_only=True)

    study = optuna.create_study()
    study.optimize(objective, n_trials=args.n_trials)

    print(f"\n--- Finished trials ---\nBest params:\n{study.best_params}---\nBest accuracy:\n{-study.best_value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
        help="data folder name")
    parser.add_argument("-n", "--n_trials", type=int, default=10, 
        help="number of trials")
    parser.add_argument("-m", "--mode", type=str, default="regression", choices=['classif', 'regression', 'density'],
        help="name of the task")
    parser.add_argument("-t", "--train_size", type=float, default=0.8, 
        help="dataset train size")
    parser.add_argument("-k", "--print_every_k_batch", type=int, default=10, 
        help="maximum number of samples for training and testing")
    parser.add_argument("-v", "--voc_path", type=str, default=None, 
        help="path to the new words to be added to the vocabulary of camembert")
    parser.add_argument("-nr", "--nrows", type=int, default=None, 
        help="maximum number of samples for training and testing")
    parser.add_argument("-e", "--epochs", type=int, default=10, 
        help="number of epochs")
    parser.add_argument("-r", "--resume", type=str, default=None, 
        help="result folder in with the saved checkpoint will be reused")
    main(parser.parse_args())