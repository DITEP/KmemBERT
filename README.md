# EHR_Transformers

Transformers for time of survival estimation based on french EHRs

## Getting started

Please make sure you have python >= 3.7. **Every** script has to be executed at the `src` folder.

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

## Preprocessing

Many scripts are available under `./src/preprocessing` and all deals with preprocessing. The preprocessing pipeline is described below.

1. `concatenate_files.py`

Get all data files on the VM and create a uniform concatenated file out of it.
The final table stored in `concatenate.txt` is separated with `£` and has 9 columns: `["Noigr","clef","Date deces","Date cr","Code nature","Nature doct","Sce","Contexte","Texte"]`

2. `split_dataset.py`

This is one of the main script. Assuming that we already have the `concatenate.txt` file, it creates three other files from it: `train.csv`, `test.csv`, `config.json`.

The data are split according to the IGR numbers, so that no IGR number in the test set can be found within the train set.
This is also where we compute the mean survival time, which is then stored in `config.json`.
We also perform some preprocessing in this script, notably:
- **Filtering**: we only keep EHRs with the type `"C.R. consultation"`. This step reduces the number of samples from 2,904,066 to 1,347,612
- **Missing Values**: we get rid of samples where one of this value is missing: `["Date deces", "Date cr", "Texte", "Noigr"]`. This step removes about 100 samples
- **Negative survival time**: some EHRs are signed or completed after the decease, we get rid of those samples. This step removes about 5k samples


3. (`visualize_data.py`)

Optional. Data visualization can be produced only after the execution of the previous scripts. 

4. `extract_unknown_words.py`

Extract unknwon words and their occurences among a whole dataset. Unknown = according to Camembert vocabulary.
It creates a json file under `./medical_voc`.

5. `preprocess_voc.py`

Modifies a medical_voc json previously created. It removes duplicates (words that exist with **and** without an `s`) and misspelled words.

6. `correction.py`

Corrects a dataset using sym_spell to remove the main misspellings and accents missings.

## Training a model

Once a clean dataset is created according to the previous section, one can train a model.
It retrains a pre-trained camembert model on a given csv dataset for a classification, regression or density task.

It creates a folder in `./results` where results are saved.

```
cd src
python training.py <command-line-arguments>
```

Execute `python training.py -h` to know more about all the possible command line parameters (see below).

```
  -h, --help            
        show this help message and exit
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
        path to the new words to be added to the vocabulary of camembert
  -r RESUME, --resume RESUME
        result folder in with the saved checkpoint will be reused
  -p PATIENCE, --patience PATIENCE
        Number of decreasing accuracy epochs to stop the training
```

For example, the following command line gets the csv files inside `./data/ehr`, sets the dropout rate to 0.5, and uses the classification mode.

```bash
python src/training.py --dataset ehr -drop 0.5 --mode classif
```

## Testing

```
cd src
python testing.py <command-line-arguments>
```

Execute `python testing.py -h` to know more about all the possible command line parameters.

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