# K-memBERT

Estimation of cancer patients survival time based on french medical reports and using Transformers

## Getting started

Please make sure you have python >= 3.7. **Every** script has to be executed at the root of the project using the `-m` option (please consider the doc about [using the module option](https://docs.python.org/3/using/cmdline.html)).

The requirements can be installed with the following command line:

```
pip install -r requirements.txt
```

Before to continue, please make sure the following command line is correctly running. If it runs until printing "DONE" then you can move on to the next section.

```
python -m kmembert.training
```

## Installing the kmembert package

If you want so, you can install the package using pip, else move on to the next section. At the root of the project, install it via pip:

`sudo pip install .`

Or in dev mode

`sudo pip install -e .`

## Project structure

The main files and folders are briefly described below. Some files that don't need to be described are not listed below.

```
.
├── data         - folder containing the csv data files (details below)
│   └── ...
├── medical_voc  - folder containing the medical vocabulary files
│   └── ...
├── results      - folder storing results (see its own README)
│   └── ...
├── graphs       - folder storing graphs and data viz
│   └── ...
└── kmembert                          - python package
    ├── models
    │   ├── conflation.py        - conflation of multiple ehrs
    │   ├── health_bert.py       - main transformer model
    │   ├── interface.py         - model class interface
    │   ├── sanity_check.py
    │   ├── time2vec.py
    │   └── transformer_aggregator.py  - transformer using multiple ehrs
    ├── preprocessing            - folder containing preprocessing scripts
    │   ├── concatenate_files.py - concatenate GR data files
    │   ├── correction.py        - correct mispellings on a dataset
    │   ├── extract_unknown_words.py  - build a medical vocabulary
    │   ├── preprocess_voc.py    - clean up the medical vocabulary
    │   ├── split_dataset.py     - splits a dataset (details below)
    │   └── visualize_data.py    - visualize a dataset
    ├── config.py                - class containing the config variables
    ├── baseline.py              - TFIDF baseline
    ├── dataset.py               - PyTorch Dataset implementation
    ├── history.py               - training of models using multiple ehrs
    ├── health_bert.py           - camembert implementation
    ├── hyperoptimization.py     - optuna hyperoptimization
    ├── training.py              - training and validation of a model
    ├── testing.py               - testing of a model
    ├── predict.py               - runs a model on a few samples
    ├── preprocesser.py          - preprocess texts
    ├── utils.py                 - utils
    └── visualize_attention.ipynb - visualize bert heads attention
```

## Data

The data are stored inside `./data`. A folder inside `./data` corresponds to one single dataset, and has to contain four files. For example, the `./data/ehr` folder has the right format.

For more details, see `./data/README.md`. 

## Preprocessing

Many scripts are available under `./kmembert/preprocessing` and all deals with preprocessing. The preprocessing pipeline is described below.

1. `concatenate_files.py`

Get all data files on the VM and create a uniform concatenated file out of it.
The final table stored in `concatenate.txt` is separated with `£` and has 9 columns: `["Noigr","clef","Date deces","Date cr","Code nature","Nature doct","Sce","Contexte","Texte"]`

2. `split_dataset.py`

This is one of the main script. Assuming that we already have the `concatenate.txt` file, it creates four other files from it: `train.csv`, `test.csv`, `config.json`, `validation_split.csv`.

The data are split according to the IGR numbers, so that no IGR number in the test set can be found within the train set. We also split the data contained in `train.csv` into a train set and validation one, again with no common patient. This split is stored in the `validation_split.csv` file and will be used for training. 
This is also where we compute the mean survival time, which is then stored in `config.json`.
We also perform some preprocessing in this script, notably:
- **Filtering**: we only keep EHRs with the type `"C.R. consultation"`. This step reduces the number of samples from 2,904,066 to 1,347,612
- **Missing Values**: we get rid of samples where one of this value is missing: `["Date deces", "Date cr", "Texte", "Noigr"]`. This step removes about 100 samples
- **Negative survival time**: some EHRs are signed or completed after the decease, we get rid of those samples. This step removes about 5k samples


3. (`visualize_data.py`)

Optional. Data visualization can be produced only after the execution of the previous scripts. Given a dataset, it will produce 6 plots highlighting : number of ehr per patient, sentence per ehr, mean token per sentence, number of tokens per ehr, number of known tokens per ehr, survival time distribution.

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

It creates a folder in `./results` where results are saved. See `./results/README.md` for more details.

```
python -m kmembert.training <command-line-arguments>
```

Execute `python -m kmembert.training -h` to know more about all the possible command line parameters (see below).

```
  -h, --help
      show this help message and exit
  -d DATA_FOLDER, --data_folder DATA_FOLDER
      data folder name
  -m {regression,density}, --mode {regression,density}
      name of the task
  -b BATCH_SIZE, --batch_size BATCH_SIZE
      dataset batch size
  -e EPOCHS, --epochs EPOCHS
      number of epochs
  -drop DROP_RATE, --drop_rate DROP_RATE
      dropout ratio. By default, None uses p=0.1
  -nr NROWS, --nrows NROWS
      maximum number of samples for training and validation
  -k PRINT_EVERY_K_BATCH, --print_every_k_batch PRINT_EVERY_K_BATCH
      prints training loss every k batch
  -f [FREEZE], --freeze [FREEZE]
      whether or not to freeze the Bert part
  -dt DAYS_THRESHOLD, --days_threshold DAYS_THRESHOLD
      days threshold to convert into classification task
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
      model learning rate
  -r_lr RATIO_LR_EMBEDDINGS, --ratio_lr_embeddings RATIO_LR_EMBEDDINGS
      the ratio applied to lr for embeddings layer
  -wg WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
      the weight decay for L2 regularization
  -v VOC_FILE, --voc_file VOC_FILE
      voc file containing camembert added vocabulary
  -r RESUME, --resume RESUME
      result folder in which the saved checkpoint will be reused
  -p PATIENCE, --patience PATIENCE
      number of decreasing accuracy epochs to stop the training
```

For example, the following command line gets the csv files inside `./data/ehr`, uses the first 10,000 rows, and uses the density mode.

```
python -m kmembert.training --data_folder ehr --nrows 10000 --mode density
```

## Other scripts

You can also execute the following scripts:
- `kmembert.preprocessing.<every-file>` runs preprocessing (every file under preprocessing can be run)
- `kmembert.history` runs a multi-ehr model
- `kmembert.testing` tests a model
- `kmembert.hyperoptimization` runs hyperoptimization
- `kmembert.baseline` runs the baseline
- `kmembert.predict` runs the model on a few samples

Using the following command line

```
python -m kmembert.<filename> <command-line-arguments>
```

Execute `python -m kmembert.<filename> -h` to know more about all the possible command line parameters.

## Testing

There is a bash script at the root of the project: `test.sh`. This script aims at testing that the main python scripts runs smoothly. It basically runs a bunch of python scripts and check that they do not return any errors.

To run the script, execute the following command:
```
bash test.sh
```
If you want a more exhaustive testing (it is going to take more time), execute the following command:
```
bash test.sh long
```
