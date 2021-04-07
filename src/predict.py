
from .utils import get_label
import numpy as np
import pandas as pd
import os
import torch
from transformers import CamembertForSequenceClassification

from models.health_bert import HealthBERT
from config import Config



def predict(texts):
    # Load the model
    config = Config({'resume': '../results/training_21-04-05_10h02m00s'})
    model = HealthBERT("cpu", config)

    # Make the prediction:
    outputs = model.step(texts)

    predictions = torch.softmax(outputs, dim=1).argmax(axis=1).tolist()
    print(predictions)

    return predictions

if __name__ == '__main__':

    text = ["Ce qui est sûr c'est que le patient va bientôt mourir"]
    predict(text)













