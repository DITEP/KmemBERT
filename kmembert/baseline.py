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
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, balanced_accuracy_score, roc_auc_score, f1_score

from .utils import get_root, get_label
from .preprocesser import EHRPreprocesser
import json

def main(args):
    # Load data and split it
    path_root = get_root()
    results = {}
    
    print('load data')
    path_data = os.path.join(path_root, "data", args.data_folder)
    df = pd.read_csv(os.path.join(path_data, "train.csv"), nrows=args.nrows)

    preprocesser = EHRPreprocesser()

    if os.path.isfile(os.path.join(path_data, "validation_split.csv")):
        validation_split = pd.read_csv(os.path.join(path_data, "validation_split.csv"), dtype=bool, nrows=args.nrows)
    
        validation = df[validation_split["validation"]]
        train = df.drop(validation.index)
        data = {"val": validation, "train":train}

        labels = {key : np.array(list(d[["Date deces", "Date cr"]].apply(lambda x: get_label(*x), axis=1))) for key, d in data.items()}
        texts = {key : list(d["Texte"].apply(preprocesser)) for key, d in data.items()}
    else :
        labels = np.array(list(df[["Date deces", "Date cr"]].apply(lambda x: get_label(*x), axis=1)))
        texts = list(df["Texte"].apply(preprocesser))
        X_train, X_val, y_train, y_val = train_test_split(texts, labels, train_size=args.train_size, random_state=0)
        labels = {"val" : y_val, "train" : y_train}
        texts = {"val" : X_val, "train" : X_train}
    print("Data loaded on {} lines".format(args.nrows))

    # Build model, train and evaluate
    assert args.model in ["RF", "MLP"], "model argument should be either RF or MLP"
    if args.model == "RF":
        ehr_regressor = Pipeline([('tfidf', TfidfVectorizer(min_df=args.min_tf, max_features=args.max_f)),
                        ('rf', RandomForestRegressor(verbose = args.verbose*1, n_jobs=-1)),
                        ], verbose=args.verbose)
    else :
        ehr_regressor = Pipeline([('tfidf', TfidfVectorizer(min_df=args.min_tf)),
                        ('mlp', MLPRegressor(hidden_layer_sizes = (200,200,),verbose = args.verbose, max_iter=800)),
                        ], verbose=args.verbose)

    print("Lauching training... {} chosen as decoder with tf idf min count {}".format(args.model, args.min_tf))
    
    print(texts['train'][:args.n_train_tfidf])
    ehr_regressor['tfidf'].fit(texts["train"][:args.n_train_tfidf])
    X_tfidf = ehr_regressor['tfidf'].transform(texts["train"])
    ehr_regressor['rf'].fit(X_tfidf, labels["train"])
    
    #ehr_regressor.fit(texts["train"], labels["train"])
    
    print("Vocabualry size : {}".format(len(ehr_regressor['tfidf'].vocabulary_)))

    predictions = ehr_regressor.predict(texts["val"])
    rmse = mean_squared_error(labels["val"], predictions)**0.5
    mae = mean_absolute_error(labels["val"], predictions)

    corr = np.corrcoef(np.array(predictions), np.array(labels["val"]))[0,1]

    d_for_accuracy = 365
    bin_predictions = (predictions >= d_for_accuracy).astype(int)
    bin_labels = (labels["val"] >= d_for_accuracy).astype(int)
    acc = balanced_accuracy_score(bin_labels, bin_predictions)
    f1 = f1_score(bin_labels, bin_predictions, average=None).tolist()
    try:
        auc = roc_auc_score(bin_labels, bin_predictions).tolist()
    except:
        auc = None

    print("RMSE validation set : {}\tMAE : {}\tBalanced Accuracy : {}\tCorrelation : {}\tF1 : {}\tAUC : {}\t".format(round(rmse,1), round(mae,1), round(acc*100,1), round(corr,4), f1, auc))

    results['RMSE'] = rmse
    results['MAE'] = mae
    results['Balanced accuracy'] = acc
    results['f1'] = f1
    results['AUC'] = auc
    results['corr'] = corr
      
    # path to savings
    path_to_save = os.path.join(path_root, args.folder_to_save)
    
    # Save the results
	    
    with open(os.path.join(path_to_save, "results.json"), "w") as fp:
        json.dump(results, fp)
	        
    # Save the model
   
    if not os.path.isdir(path_to_save):
        os.mkdir(path_to_save)
    model_name = "model_mae" + str(round(mae,1))+".pkl"
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
    parser.add_argument("-m", "--model", type=str, default="RF",
        help = "Model to use for decoding : RF, MLP")
    parser.add_argument("-mtf", "--min_tf", type=int, default=50,
        help = "Minimum number of count for a word to be taken into account in tf idf")
    parser.add_argument("-mf", "--max_f", type=int,default=None, 
	help = "Maximum term frequency across the corpus")
    parser.add_argument("--n_train_tfidf", type=int, default=50000,
        help = "Number of reports used to train TFIDF")
    
    main(parser.parse_args())