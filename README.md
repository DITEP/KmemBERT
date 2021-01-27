# EHR_Transformers

Transformers for time of survival estimation based on french EHRs

## Getting started

The environment can be installed using conda.

```bash
conda env create -f environment.yml
```

It creates a `gr` environment.

On this environment and at the root of the project, make sure the following command line is correctly running.

```bash
python src/training.py
```

## Project structure

The main files and folders are briefly described bellow. Some files that don't need to be described are not listed bellow.

```bash
.
├── data                         - folder of csv data files
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

## Training a model

Retrain a pre-trained camembert model on a given csv dataset for a classification or regression task.

It creates a folder in `./results` where results are saved.

```bash
python src/training.py <command-line-arguments>
```

Execute `python src/training.py -h` to know more about all the possible command line parameters (e.g. see bellow).

```bash
optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        dataset filename
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

For example, the following command line used the given csv input file, set the dropout rate to 0.5, and use the classification mode.

```bash
python src/training.py --dataset file_name.csv -drop 0.5 --classify
```

## Fine tuning hyperparameters

Fine tuning of hyperparameters using Optuna.

```bash
python src/hyperoptimization.py <command-line-arguments>
```

Execute `python src/hyperoptimization.py -h` to know more about all the possible command line parameters.

## Saved results

This is an example of a result folder saved after training.

```bash
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

The checkpoint is composed of the following items.
```bash
{
    'model': model.state_dict(),   - model state dict
    'accuracy': test_accuracy,     - model accuracy on the test dataset
    'epoch': epoch,                - at which epoch it was saved
    'tokenizer': model.tokenizer   - the model tokenizer
}
```