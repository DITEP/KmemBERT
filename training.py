import numpy as np
import os
import argparse
import json

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import CamembertForSequenceClassification, pipeline, CamembertTokenizer

from dataset import TweetDataset
from utils import get_root

def main(batch_size, epochs, train_size):
    path_root = get_root()
    print("PATH_ROOT:", path_root)

    csv_path = os.path.join(path_root, "french_tweets_short.csv")
    model_name = "camembert-base"
    save_model_path = os.path.join(path_root, "camembert_model")

    dataset = TweetDataset(csv_path)
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    camembert = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    optimizer = Adam(camembert.parameters(), lr=1e-4)
    tokenizer = CamembertTokenizer.from_pretrained(model_name)
    classifier = pipeline('sentiment-analysis', model=camembert, tokenizer=tokenizer)

    # Train
    camembert.train()
    for epoch in range(epochs):
        train_loss = 0
        for texts, labels in train_loader:
            encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            optimizer.zero_grad()
            
            output = camembert(input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
    print("----- Ended Training\n")
    torch.save(camembert, save_model_path)
    print("Model saved")

    # Test
    predictions, test_labels = [], []
    for texts, labels in test_loader:
        encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        output = camembert(input_ids, attention_mask=attention_mask)

        predictions += torch.softmax(output.logits, dim=1).argmax(axis=1).tolist()
        test_labels += labels.tolist()

    print("Labels:\n", test_labels)
    print("Predictions:\n", predictions)
    print(f"Test accuracy: {1 - np.mean(np.abs(np.array(test_labels)-np.array(predictions)))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=64, 
        help="dataset batch size")
    parser.add_argument("-e", "--epochs", type=int, default=2, 
        help="number of epochs")
    parser.add_argument("-t", "--train_size", type=float, default=0.8, 
        help="dataset train size")
    args = parser.parse_args()
    print(f"\n> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")
    
    main(args.batch_size, args.epochs, args.train_size)