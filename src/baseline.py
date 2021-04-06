'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import argparse
import os
import pandas as pd 
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .utils import get_root, get_label
from .preprocesser import EHRPreprocesser


def main(args):
    # Load data and split it
    path_root = get_root()
    path_dataset = os.path.join(path_root, "data", args.data_folder, "train.csv")
    df = pd.read_csv(path_dataset, nrows=args.nrows)
    labels = np.array(list(df[["Date deces", "Date cr"]].apply(lambda x: get_label(*x), axis=1)))
    preprocesser = EHRPreprocesser()
    texts = list(df["Texte"].apply(preprocesser))
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, train_size=args.train_size, random_state=0)

    # Build model, train and evaluate
    ehr_regressor = Pipeline([('tfidf', TfidfVectorizer()),
                      ('rf', RandomForestRegressor()),
                    ], verbose=args.verbose)

    ehr_regressor.fit(X_train, y_train)

    predictions = ehr_regressor.predict(X_val)
    rmse = mean_squared_error(y_val, predictions)**0.5
    mae = mean_absolute_error(y_val, predictions)

    print("RMSE validation set : {}    MAE : {}".format(round(rmse,1), round(mae,1)))

    # Save the model
    path_to_save = os.path.join(path_root, args.folder_to_save)
    if not os.path.isdir(path_to_save):
        os.mkdir(path_to_save)
    joblib.dump(ehr_regressor, os.path.join(path_to_save, "model.pkl"), compress = 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
        help="data folder name")
    parser.add_argument("-t", "--train_size", type=float, default=0.8, 
        help="dataset train size")
    parser.add_argument("-nr", "--nrows", type=int, default=None, 
        help="maximum number of samples for training and testing")
    parser.add_argument("-fs", "--folder_to_save", type=str, default="baseline",
    help = "folder to save the model")
    parser.add_argument("-v", "--verbose", type=bool, default=True,
    help = "verbose arg of the pipeline")


    main(parser.parse_args())