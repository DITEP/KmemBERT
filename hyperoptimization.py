import argparse
import json
import os

import optuna
import torch
from torch.utils.data import DataLoader

from utils import get_root
from training import train_and_test
from dataset import TweetDataset


def main(args):
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:",device)
    path_root = get_root()
    print("PATH_ROOT:", path_root)

    csv_path = os.path.join(path_root, args.dataset)
    model_name = "camembert-base"

    dataset = TweetDataset(csv_path)
    train_size = min(args.max_size, int(args.train_size * len(dataset)))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    def objective(trial):
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-4)
        epochs = trial.suggest_int('epochs', 3, 4)
        freeze = trial.suggest_categorical('freeze', [False, True])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return -train_and_test(train_loader, test_loader, device, args.voc_path, model_name, args.classify, args.print_every_k_batch, args.max_size,
                   batch_size, learning_rate, epochs, freeze)

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
    args = parser.parse_args()
    print(f"\n> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")
    
    main(args)