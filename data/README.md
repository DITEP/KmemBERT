# Data

The data is stored inside this folder. A folder corresponds to one single dataset, and has to contain three files. For example, the `ehr` dataset bellow has the right format.

> NB: the `ehr` dataset is just a fake example with the right format

```
.
└── ehr                         - one dataset
    ├── config.json             - config file
    ├── test.csv                - test set
    ├── train.csv               - train set
    └── validation_split.csv    - validation split (optional)
```

## Train and Test csv

You will need a `train.csv` file to train a model, and a `test.csv` to test it. If one is missing, you will be able to run only one of the two previous scripts.

The CSV separator has to be a comma.

### Columns

The required columns are:
- Date deces
- Date cr
- Texte

For example,
```
Noigr,clef,Date deces,Date cr,Code nature,Nature doct,Sce,Contexte,Texte
```
## Config file

The config file has to be named `config.json`.

> It is automatically created during the data processing.

This is a config file example.
```
{
    "mean_time_survival": 800
}
```

## Validation split 
This is a data frame of one column filled with booleans indicating the row to use for validation set in the `train.csv` file. In practice, we make sure that there isn't any patient with EHR both in training and validation. 

> It is automatically created during the data processing.
