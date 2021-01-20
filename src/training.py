import numpy as np
import os
import argparse
import json
from time import time
from collections import defaultdict
from numpy.lib.function_base import average
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from dataset import TweetDataset
from utils import get_root, pretty_time, printc, create_session, save_json
from health_bert import HealthBERT

def train_and_test(train_loader, test_loader, device, voc_path, model_name, classify, print_every_k_batch, max_size,
                   batch_size, learning_rate, epochs, freeze, path_result, ratio_lr_embeddings=1):
    """
    Creates a camembert model and retrain it, with eventually a larger vocabulary.

    Inputs: please refer bellow, to the argparse arguments.
    """

    model = HealthBERT(device, 
                       learning_rate, 
                       voc_path=voc_path, 
                       model_name=model_name, 
                       classify=classify, 
                       freeze=freeze, 
                       ratio_lr=ratio_lr_embeddings)

    # Train
    model.train()

    printc("\n----- STARTING TRAINING -----")
    losses = defaultdict(list)
    n_samples = print_every_k_batch * batch_size

    for epoch in range(epochs):
        print("> EPOCH", epoch)
        epoch_loss, k_batch_loss = 0, 0
        epoch_start_time, k_batch_start_time = time(), time()
        model.start_epoch_timers()
        for i, (texts, labels) in enumerate(train_loader):
            loss, _ = model.step(texts, labels)

            epoch_loss += loss.item()
            k_batch_loss += loss.item()

            if (i+1) % print_every_k_batch == 0:
                average_loss = k_batch_loss / n_samples
                print('    [{}-{}]  -  Average loss: {:.4f}  -  Time elapsed: {} - Time encoding: {} - Time forward: {}'.format(
                    i+1-print_every_k_batch, i+1, 
                    average_loss, 
                    pretty_time(time()-k_batch_start_time), 
                    pretty_time(model.encoding_time), 
                    pretty_time(model.compute_time)
                ))
                losses[epoch].append(average_loss)
                k_batch_loss = 0
                k_batch_start_time = time()
        printc(f'    Global average loss: {epoch_loss/len(train_loader.dataset):.4f}  -  Time elapsed: {pretty_time(time()-epoch_start_time)}\n', 'RESULTS')
    
    printc("-----  Ended Training  -----\n")
    save_json(path_result, "loss", losses)
    plt.plot(np.linspace(0, epochs, sum([len(l) for l in losses.values()])),
             [ l for ll in losses.values() for l in ll ])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(os.path.join(path_result, "loss.png"))
    #torch.save(model, os.path.join(path_result, "model"))
    #printc("Model saved", "SUCCESS")

    # Test
    model.eval()
    predictions, test_labels = [], []
    test_start_time = time()
    for i, (texts, labels) in enumerate(test_loader):
        loss, logits = model.step(texts, labels)

        predictions += torch.softmax(logits, dim=1).argmax(axis=1).tolist()
        test_labels += labels.tolist()

        if(i*batch_size > max_size):
            break
    save_json(path_result, "test", {"labels": test_labels, "predictions": predictions})
    test_accuracy = 1 - np.mean(np.abs(np.array(test_labels)-np.array(predictions)))
    printc(f"\n> Test accuracy: {test_accuracy}", 'RESULTS')
    printc(f"> Test time: {pretty_time(time()-test_start_time)}", 'RESULTS')
    return test_accuracy

def main(args):
    path_root, path_result, device = create_session(args)

    csv_path = os.path.join(path_root, "data", args.dataset)
    model_name = "camembert-base"

    dataset = TweetDataset(csv_path)
    train_size = min(args.max_size, int(args.train_size * len(dataset)))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    train_and_test(train_loader, test_loader, device, args.voc_path, model_name, args.classify, args.print_every_k_batch, args.max_size,
                   args.batch_size, args.learning_rate, args.epochs, args.freeze, path_result, args.ratio_lr_embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="french_tweets_short.csv", 
        help="dataset filename")
    parser.add_argument("-c", "--classify", type=bool, default=False, const=True, nargs="?",
        help="whether or not to train camembert for a classification task")
    parser.add_argument("-b", "--batch_size", type=int, default=8, 
        help="dataset batch size")
    parser.add_argument("-e", "--epochs", type=int, default=2, 
        help="number of epochs")
    parser.add_argument("-t", "--train_size", type=float, default=0.8, 
        help="dataset train size")
    parser.add_argument("-max", "--max_size", type=int, default=10000, 
        help="maximum number of samples for training and testing")
    parser.add_argument("-k", "--print_every_k_batch", type=int, default=1, 
        help="maximum number of samples for training and testing")
    parser.add_argument("-f", "--freeze", type=bool, default=False, const=True, nargs="?",
        help="whether or not to freeze the Bert part")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, 
        help="dataset train size")
    parser.add_argument("-r_lr", "--ratio_lr_embeddings", type=float, default=1, 
        help="the ratio applied to lr for embeddings layer")
    parser.add_argument("-v", "--voc_path", type=str, default=None, 
        help="path to the new words to be added to the vocabulary of camembert")

    main(parser.parse_args())