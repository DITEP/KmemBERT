# EHR_Transformers

Transformers for time of survival estimation based on french EHRs

## Getting started

Please make sure you have python >= 3.7. Every script has to be executed at the `src` folder.

The requirements can be installed with the following command line:

```
pip install -r requirements.txt
```

Before to continue, please make sure the following command line is correctly running. If if runs until printing "DONE" then you can move on to the next section.

```
cd src
python training.py
```

## Project structure

The main files and folders are briefly described below. Some files that don't need to be described are not listed below.

```
.
├── data                         - folder containing the csv data files (details below)
├── exploration                  - folder of notebooks used for exploration
├── medical_voc                  - folder containing the medical vocabulary
├── results                      - folder storing results
├── src                          - python package
│   ├── preprocessing            - folder containing preprocessing scripts
│   │   ├── concatenate_files.py - concatenate GR data files
│   │   ├── correction.py        - correct mispellings on a dataset
│   │   ├── extract_unknown_words.py  - build a medical vocabulary
│   │   ├── preprocess_voc.py    - clean up the medical vocabulary
│   │   ├── split_dataset.py     - split a dataset into train and test and apply basic preprocessing (details below)
│   │   └── visualize_data.py    - visualize a dataset
│   ├── config.py                - class containing the config variables
│   ├── dataset.py               - PyTorch Dataset implementation
│   ├── health_bert.py           - camembert implementation
│   ├── hyperoptimization.py     - optuna hyperoptimization
│   ├── training.py              - training and validation of a model
│   ├── testing.py               - testing of a model
│   └── utils.py                 - utils
└── ****.sh                      - scripts used to run a job on the cluster of CentraleSupélec
```

## Data

The data are stored inside `./data`. A folder inside `./data` corresponds to one single dataset, and has to contain three files. For example, the `ehr` dataset below has the right format.

```
.
└── data                 - data folder
    └── ehr              - one dataset
        ├── config.json  - config file
        ├── test.csv     - test set
        └── train.csv    - train set
```

For more details, please refer to `./data/README.md`. 

## Training a model

Retrain a pre-trained camembert model on a given dataset for a classification or regression task.

It creates a folder in `./results` where results are saved.

```
cd src
python training.py <command-line-arguments>
```

Execute `python training.py -h` to know more about all the possible command line parameters (see below).

```
  -h, --help            show this help message and exit
  -d DATA_FOLDER, --data_folder DATA_FOLDER
                        data folder name
  -m {classif,regression,density}, --mode {classif,regression,density}
                        name of the task
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        dataset batch size
  -e EPOCHS, --epochs EPOCHS
                        number of epochs
  -t TRAIN_SIZE, --train_size TRAIN_SIZE
                        dataset train size
  -drop DROP_RATE, --drop_rate DROP_RATE
                        dropout ratio
  -nr NROWS, --nrows NROWS
                        maximum number of samples for training and testing
  -k PRINT_EVERY_K_BATCH, --print_every_k_batch PRINT_EVERY_K_BATCH
                        maximum number of samples for training and testing
  -f [FREEZE], --freeze [FREEZE]
                        whether or not to freeze the Bert part
  -dt DAYS_THRESHOLD, --days_threshold DAYS_THRESHOLD
                        days threshold to convert into classification task
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        dataset train size
  -r_lr RATIO_LR_EMBEDDINGS, --ratio_lr_embeddings RATIO_LR_EMBEDDINGS
                        the ratio applied to lr for embeddings layer
  -wg WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
                        the weight decay for L2 regularization
  -v VOC_PATH, --voc_path VOC_PATH
                        path to the new words to be added to the vocabulary of
                        camembert
  -r RESUME, --resume RESUME
                        result folder in with the saved checkpoint will be
                        reused
  -p PATIENCE, --patience PATIENCE
                        Number of decreasing accuracy epochs to stop the
                        training
```

For example, the following command line gets the csv files inside `./data/ehr`, sets the dropout rate to 0.5, and uses the classification mode.

```bash
python src/training.py --dataset ehr -drop 0.5 --mode classif
```

## Fine tuning hyperparameters

Fine tuning of hyperparameters using Optuna.

```
cd src
python hyperoptimization.py <command-line-arguments>
```

Execute `python hyperoptimization.py -h` to know more about all the possible command line parameters.

## Saved results

This is an example of a result folder saved after training.

```
.
└── training_21-01-27_16h25m41s - folder name (script name + date)
    ├── args.json               - command line arguments
    ├── checkpoint.pth          - checkpoint (see below)
    ├── loss.png                - train and validation loss evolution
    ├── losses.json             - json of all the losses
    └── test.json               - predictions and labels on the validation dataset
```

### Checkpoint

Use `checkpoint = torch.load(<path_checkpoint>, map_location=<device>)` to load a checkpoint on a given device.

The checkpoints are composed of the following items.
```
{
    'model': model.state_dict(),   - model state dict
    'accuracy': test_accuracy,     - model accuracy on the validation dataset
    'epoch': epoch,                - at which epoch it was saved
    'tokenizer': model.tokenizer   - the model tokenizer
}
```