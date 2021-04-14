'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import numpy as np
import pandas as pd
import argparse
from lime.lime_text import LimeTextExplainer

from .models import HealthBERT
from .config import Config
from .utils import create_session




def predict(texts):
    outputs = model.step(texts)
    predictions = []
    for i in range(len(outputs)):
      valeur = outputs[i].item()
      predictions.append([1-valeur, valeur])

    return np.array(predictions)

def main(args):
    global model
    _, _, device, config = create_session(args)
    #config = Config(args)
    # config.path_result = ""
    # config.resume = "training_21-04-05_10h02m00s"

    model = HealthBERT("cpu", config)
    #file_to_classify = pd.read_csv(config.data_path, nrows = config.nrows)
    #text_to_classify = file_to_classify.Texte

    text_to_classify = ["il va bientôt mourir", "Le patient va très bien, son corps se comporte bien, il va bientôt guérir","Aujourd'hui, il y a eu une très grande amélioration de l'état du patient","Le patient va mourir en moins d'une semaine, c'est alertant"]
    class_names = ["Moins de trois mois", "Plus de trois mois"]

    explainer = LimeTextExplainer(class_names=class_names)

    exp = explainer.explain_instance(text_to_classify[0], predict, num_features=4)
    print(exp.as_list())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
        help="data path to access to the testing file")
    parser.add_argument("-r", "--resume", type=str, required=True, 
        help="result folder in with the saved checkpoint will be reused")
    
    parser.add_argument("-nr", "--nrows", type=int, default=None, 
        help="maximum number of samples for testing")

    main(parser.parse_args())













