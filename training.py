import numpy as np
import os
import argparse
import json
from time import time

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import CamembertForSequenceClassification, pipeline, CamembertTokenizer

from dataset import TweetDataset
from utils import get_root, pretty_time

def main(dataset, batch_size, epochs, train_size, max_size, print_every_k_batch, freeze):
    torch.manual_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:",device)

    path_root = get_root()
    print("PATH_ROOT:", path_root)

    csv_path = os.path.join(path_root, dataset)
    model_name = "camembert-base"
    save_model_path = os.path.join(path_root, "camembert_model")

    dataset = TweetDataset(csv_path)
    train_size = min(max_size, int(train_size * len(dataset)))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    camembert = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    camembert.to(device)
    if(freeze):
        for param in camembert.roberta.parameters():
            param.requires_grad = False

    optimizer = Adam(camembert.parameters(), lr=1e-4)
    tokenizer = CamembertTokenizer.from_pretrained(model_name)
    classifier = pipeline('sentiment-analysis', model=camembert, tokenizer=tokenizer)

    # Train
    camembert.train()
    for epoch in range(epochs):
        epoch_loss, k_batch_loss = 0,0
        epoch_start_time, k_batch_start_time = time(), time()
        for i, (texts, labels) in enumerate(train_loader):
            encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = camembert(input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss

            loss.backward()
            epoch_loss += loss.item()
            k_batch_loss += loss.item()
            optimizer.step()

            if((i+1)%print_every_k_batch == 0):
                # TODO: more precise
                n_samples = print_every_k_batch * batch_size
                print('{}> Epoch {} Batches [{}-{}]  -  Average loss: {:.4f}  -  Time elapsed: {}'.format(
                    "=" * ((i+1)//print_every_k_batch),
                    epoch, i+1-print_every_k_batch, i+1, k_batch_loss / n_samples, pretty_time(time()-k_batch_start_time)))
                k_batch_loss = 0
                k_batch_start_time = time()

        print('> Epoch: {}  -  Global average loss: {:.4f}  -  Time elapsed: {}\n'.format(
            epoch, epoch_loss / len(train_loader.dataset), pretty_time(time()-epoch_start_time)))
    print("----- Ended Training\n")
    torch.save(camembert, save_model_path)
    print("Model saved")

    # Test
    predictions, test_labels = [], []
    test_start_time = time()
    for i, (texts, labels) in enumerate(test_loader):
        encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        output = camembert(input_ids, attention_mask=attention_mask)

        predictions += torch.softmax(output.logits, dim=1).argmax(axis=1).tolist()
        test_labels += labels.tolist()

        if(i*batch_size>max_size):
            break

    print(f"\n> Test accuracy: {1 - np.mean(np.abs(np.array(test_labels)-np.array(predictions)))}")
    print(f"> Test time: {pretty_time(time()-test_start_time)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="french_tweets_short.csv", 
        help="dataset filename")
    parser.add_argument("-b", "--batch_size", type=int, default=64, 
        help="dataset batch size")
    parser.add_argument("-e", "--epochs", type=int, default=2, 
        help="number of epochs")
    parser.add_argument("-t", "--train_size", type=float, default=0.8, 
        help="dataset train size")
    parser.add_argument("-max", "--max_size", type=int, default=10000, 
        help="maximum number of samples for training and testing")
    parser.add_argument("-k", "--print_every_k_batch", type=int, default=10, 
        help="maximum number of samples for training and testing")
    parser.add_argument("-f", "--freeze", type=bool, default=False, const=True, nargs="?",
        help="whether or not to freeze the Bert part")
    args = parser.parse_args()
    print(f"\n> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")
    
    main(args.dataset, args.batch_size, args.epochs, args.train_size, args.max_size, args.print_every_k_batch, args.freeze)