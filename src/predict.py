'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import numpy as np
from lime.lime_text import LimeTextExplainer

from .models import HealthBERT
from .config import Config



def predict(texts):
    outputs = model.step(texts)
    predictions = []
    for i in range(len(outputs)):
      valeur = outputs[i].item()
      predictions.append([1-valeur, valeur])

    return np.array(predictions)

if __name__ == '__main__':
    config = Config()
    config.path_result = ""
    config.resume = "training_21-04-05_10h02m00s"

    model = HealthBERT("cpu", config)

    text = ["il va bientôt mourir", "Le patient va très bien, son corps se comporte bien, il va bientôt guérir","Aujourd'hui, il y a eu une très grande amélioration de l'état du patient","Le patient va mourir en moins d'une semaine, c'est alertant"]
    class_names = ["Moins de trois mois", "Plus de trois mois"]

    explainer = LimeTextExplainer(class_names=class_names)

    exp = explainer.explain_instance(text[0], predict, num_features=1)
    print(exp.as_list())













