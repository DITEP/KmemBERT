import argparse
import json
import os

import optuna
import torch
from torch.utils.data import DataLoader

from utils import printc, create_session
from training import train_and_test
from dataset import TweetDataset


def main(args):
    path_dataset, path_result, device = create_session(args)

    model_name = "camembert-base"

    dataset = TweetDataset(path_dataset)
    train_size = min(args.max_size, int(args.train_size * len(dataset)))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    def objective(trial):
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-4)
        freeze = trial.suggest_categorical('freeze', [False, True])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return -train_and_test(train_loader, test_loader, device, args.voc_path, model_name, args.classify, args.print_every_k_batch, args.max_size,
                   batch_size, learning_rate, args.epochs, freeze, path_result)

    study = optuna.create_study()
    study.optimize(objective, n_trials=args.n_trials)

    print(f"\n--- Finished trials ---\nBest params:\n{study.best_params}---\nBest accuracy:\n{-study.best_value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="french_tweets_short.csv", 
        help="dataset filename")
    parser.add_argument("-n", "--n_trials", type=int, default=10, 
        help="number of trials")
    parser.add_argument("-c", "--classify", type=bool, default=False, const=True, nargs="?",
        help="whether or not to train camembert for a classification task")
    parser.add_argument("-t", "--train_size", type=float, default=0.8, 
        help="dataset train size")
    parser.add_argument("-k", "--print_every_k_batch", type=int, default=10, 
        help="maximum number of samples for training and testing")
    parser.add_argument("-v", "--voc_path", type=str, default=None, 
        help="path to the new words to be added to the vocabulary of camembert")
    parser.add_argument("-max", "--max_size", type=int, default=10000, 
        help="maximum number of samples for training and testing")
    parser.add_argument("-e", "--epochs", type=int, default=10, 
        help="number of epochs")
    main(parser.parse_args())