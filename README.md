# French-Transformers

## Training a model

`python src/training.py [-d DATASET] [-c [CLASSIFY]] [other parameters]`

Retrain a pre-trained camembert model on a given csv dataset for a classification or regression task.

Execute `python src/training.py -h` to know about all possible command line parameters.

## Fine tuning hyperparameters

`python src/hyperoptimization.py [-d DATASET] [other parameters]`

Fine tuning of hyperparameters.

Execute `python src/hyperoptimization.py -h` to know about all possible command line parameters.