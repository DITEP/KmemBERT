# EHR_Transformers

Transformers for time of survival estimation based on french EHRs

## Getting started

The environment can be installed using pip.

```bash
pip install -r requirements.txt
```

Before to continue, please make sure the following command line is correctly running. If if runs until printing "DONE" then you can move on to the next section.

```bash
python src/training.py
```

## Project structure

The main files and folders are briefly described bellow. Some files that don't need to be described are not listed bellow.

```
.
├── data                         - folder of csv data files (details bellow)
├── exploration                  - folder of notebooks used for exploration
├── medical_voc                  - folder containing jsons of vocabulary
├── results                      - folder storing results
├── sentence_piece               - training a new sentencepiece
├── src                          - python package
│   ├── config.py                - class containing config variables
│   ├── correction.py            - corrects misplellings on a dataset
│   ├── dataset.py               - PyTorch Dataset implementation
│   ├── extract_unknown_words.py - extracts unknown words from a dataset
│   ├── health_bert.py           - camembert EHR implementation
│   ├── hyperoptimization.py     - optuna hyperoptimization
│   ├── training.py              - training and testing a model
│   └── utils.py                 - utils
└── ****.sh                      - scripts used to run a job on the cluster of centralesupelec
```

## Data

The data is stored inside `./data`. A folder inside `./data` corresponds to one single dataset, and has to contain three files. For example, the `ehr` dataset bellow has the right format.

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

Retrain a pre-trained camembert model on a given csv dataset for a classification or regression task.

It creates a folder in `./results` where results are saved.

```bash
python src/training.py <command-line-arguments>
```

Execute `python src/training.py -h` to know more about all the possible command line parameters (e.g. see bellow).

```
optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --data_folder DATA_FOLDER
                        data folder name (inside /data)
  -c [CLASSIFY], --classify [CLASSIFY]
                        whether or not to train camembert for a classification task
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        dataset batch size
  -e EPOCHS, --epochs EPOCHS
                        number of epochs
  -t TRAIN_SIZE, --train_size TRAIN_SIZE
                        dataset train size
  -drop DROP_RATE, --drop_rate DROP_RATE
                        dropout ratio
  -max MAX_SIZE, --max_size MAX_SIZE
                        maximum number of samples for training and testing
  -k PRINT_EVERY_K_BATCH, --print_every_k_batch PRINT_EVERY_K_BATCH
                        maximum number of samples for training and testing
  -f [FREEZE], --freeze [FREEZE]
                        whether or not to freeze the Bert part
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
```

For example, the following command line gets the csv files inside `./data/ehr`, set the dropout rate to 0.5, and use the classification mode.

```bash
python src/training.py --dataset ehr -drop 0.5 --classify
```

## Fine tuning hyperparameters

Fine tuning of hyperparameters using Optuna.

```bash
python src/hyperoptimization.py <command-line-arguments>
```

Execute `python src/hyperoptimization.py -h` to know more about all the possible command line parameters.

## Saved results

This is an example of a result folder saved after training.

```
.
└── training_21-01-27_16h25m41s - folder name (script name + date)
    ├── args.json               - command line arguments
    ├── checkpoint.pth          - checkpoint (see bellow)
    ├── loss.png                - train and test loss evolution
    ├── losses.json             - json of all the losses
    └── test.json               - predictions and labels on the test dataset
```

### Checkpoint

Use `checkpoint = torch.load(<path_checkpoint>, map_location=<device>)` to load a checkpoint on a given device.

The checkpoints are composed of the following items.
```
{
    'model': model.state_dict(),   - model state dict
    'accuracy': test_accuracy,     - model accuracy on the test dataset
    'epoch': epoch,                - at which epoch it was saved
    'tokenizer': model.tokenizer   - the model tokenizer
}
```