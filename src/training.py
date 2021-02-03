import numpy as np
import os
import argparse
from time import time
from collections import defaultdict
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from dataset import TweetDataset
from utils import pretty_time, printc, create_session, save_json
from health_bert import HealthBERT

def test(model, test_loader, config, epoch=-1, test_losses=None):
    """
    Tests a model on a test_loader and compute its accuracy
    """
    
    model.eval()
    predictions, test_labels = [], []
    test_start_time = time()

    total_loss = 0
    for i, (texts, labels) in enumerate(test_loader):
        loss, outputs = model.step(texts, labels)
        
        if model.classify:
            predictions += torch.softmax(outputs, dim=1).argmax(axis=1).tolist()
        else:
            predictions += outputs.flatten().tolist()
        
        test_labels += labels.tolist()
        total_loss += loss.item()

        if(i*config.batch_size > config.max_size):
            break
    if test_losses is not None:
        test_losses.append(total_loss/len(test_loader))

    test_accuracy = 1 - np.mean(np.abs(np.array(test_labels)-np.array(predictions)))
    printc(f"\n    Test accuracy: {test_accuracy}  -  Time elapsed: {pretty_time(time()-test_start_time)}", 'RESULTS')

    if test_accuracy > model.best_acc:
        model.best_acc = test_accuracy
        printc("    Best accuracy so far", "SUCCESS")
        print('    Saving predictions...')
        save_json(config.path_result, "test", {"labels": test_labels, "predictions": predictions})
        print('    Saving model state...\n')
        state = {
            'model': model.state_dict(),
            'accuracy': test_accuracy,
            'epoch': epoch,
            'tokenizer': model.tokenizer
        }
        torch.save(state, os.path.join(config.path_result, './checkpoint.pth'))
        model.early_stopping = 0
    else : 
        model.early_stopping += 1
    


    return test_accuracy

def train_and_test(train_loader, test_loader, device, config, path_result):
    """
    Creates a camembert model and retrain it, with eventually a larger vocabulary.

    Inputs: please refer bellow, to the argparse arguments.
    """
    model = HealthBERT(device, config)
    if config.resume:
        printc(f"Resuming with model at {config.resume}", "INFO")
        path_checkpoint = os.path.join(os.path.dirname(config.path_result), config.resume, 'checkpoint.pth')
        assert os.path.isfile(path_checkpoint), 'Error: no checkpoint found!'
        checkpoint = torch.load(path_checkpoint, map_location=model.device)
        model.load_state_dict(checkpoint['model'])
        model.tokenizer = checkpoint['tokenizer']


    printc("\n----- STARTING TRAINING -----")

    losses = defaultdict(list)
    test_losses = []
    n_samples = config.print_every_k_batch * config.batch_size

    for epoch in range(config.epochs):
        print("> EPOCH", epoch)
        model.train()
        epoch_loss, k_batch_loss = 0, 0
        epoch_start_time, k_batch_start_time = time(), time()
        model.start_epoch_timers()

        for i, (texts, labels) in enumerate(train_loader):
            loss, _ = model.step(texts, labels)

            epoch_loss += loss.item()
            k_batch_loss += loss.item()

            if (i+1) % config.print_every_k_batch == 0:
                average_loss = k_batch_loss / n_samples
                print('    [{}-{}]  -  Average loss: {:.4f}  -  Time elapsed: {} - Time encoding: {} - Time forward: {}'.format(
                    i+1-config.print_every_k_batch, i+1, 
                    average_loss, 
                    pretty_time(time()-k_batch_start_time), 
                    pretty_time(model.encoding_time), 
                    pretty_time(model.compute_time)
                ))
                losses[epoch].append(average_loss)
                k_batch_loss = 0
                k_batch_start_time = time()
        printc(f'    Global average loss: {epoch_loss/len(train_loader.dataset):.4f}  -  Time elapsed: {pretty_time(time()-epoch_start_time)}\n', 'RESULTS')
        test(model, test_loader, config, epoch=epoch, test_losses=test_losses)
        if (config.patience is not None) and (model.early_stopping >= config.patience):
            break
    
    printc("-----  Ended Training  -----\n")

    print("Saving losses...")
    save_json(path_result, "losses", { "train": losses, "test": test_losses })
    plt.plot(np.linspace(0, config.epochs, sum([len(l) for l in losses.values()])),
             [ l for ll in losses.values() for l in ll ])
    plt.plot(test_losses)
    plt.legend(["Train loss", "Test loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss evolution")
    plt.savefig(os.path.join(path_result, "loss.png"))
    print("[DONE]")

    return model.best_acc

def main(args):
    path_dataset, path_result, device, config = create_session(args)

    dataset = TweetDataset(path_dataset)
    train_size = min(config.max_size, int(config.train_size * len(dataset)))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    train_and_test(train_loader, test_loader, device, config, path_result)

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
    parser.add_argument("-drop", "--drop_rate", type=float, default=None, 
        help="dropout ratio")
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
    parser.add_argument("-wg", "--weight_decay", type=float, default=0, 
        help="the weight decay for L2 regularization")
    parser.add_argument("-v", "--voc_path", type=str, default=None, 
        help="path to the new words to be added to the vocabulary of camembert")
    parser.add_argument("-r", "--resume", type=str, default=None, 
        help="result folder in with the saved checkpoint will be reused")
    parser.add_argument("-p", "--patience", type=int, default=4, 
        help="Number of decreasing accuracy epochs to stop the training")

    main(parser.parse_args())