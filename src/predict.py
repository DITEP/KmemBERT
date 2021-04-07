

import numpy as np
import pandas as pd
import os
import torch
from lime import lime_text
from lime.lime_text import LimeTextExplainer



from src.models.health_bert import HealthBERT
from src.config import Config



def predict(texts):
    # Load the model
    




    # Make the prediction:
    outputs = model.step(texts)
    predictions = []
    for i in range(len(outputs)):
      valeur = outputs[i].item()
      predictions.append([1-valeur, valeur])
      
    #predictions = torch.softmax(outputs, dim=1).argmax(axis=1).tolist()
    
    #print(predictions)

    return np.array(predictions)

if __name__ == '__main__':
    config = Config({})
    config.path_result = ""
    config.resume = "/training_21-04-05_10h02m00s"
    print(config.resume)
    model = HealthBERT("cpu", config)

    text = ["il va bientôt mourir", "Le patient va très bien, son corps se comporte bien, il va bientôt guérir","Aujourd'hui, il y a eu une très grande amélioration de l'état du patient","Le patient va mourir en moins d'une semaine, c'est alertant"]
    #predict([text[0]])
    class_names = ["Moins de trois mois", "Plus de trois mois"]
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text[0], predict, num_features=1)
    #print(exp)
    print(exp.as_list())













