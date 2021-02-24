# Data

The data is stored inside this folder. A folder corresponds to one single dataset, and has to contain three files. For example, the `ehr` dataset bellow has the right format.

```
.
└── ehr              - one dataset
    ├── config.json  - config file
    ├── test.csv     - test set
    └── train.csv    - train set
```

## Train and Test csv

You will need a `train.csv` file to train a model, and a `test.csv` to test it. If one is missing, you will be able to run only one of the two previous scripts.

The CSV separator has to be a comma.

### Columns

The required columns are:
- Data deces
- Date cr
- Texte

For example,
```
Noigr,clef,Date deces,Date cr,Code nature,Nature doct,Sce,Contexte,Texte
```
## Config file

The config file has to be named `config.json`. Do not create it manually, as it will be automatically generated the first time to launch a training.

This is a config file example.
```
{
    "mean_time_survival": 800
}
```