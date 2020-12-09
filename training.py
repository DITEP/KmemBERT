import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import CamembertForSequenceClassification, pipeline, CamembertTokenizer

from dataset import TweetDataset

batch_size = 64
epochs = 10
train_size = 0.7

model_name = "camembert-base"
dataset = TweetDataset("french_tweets_short.csv")

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