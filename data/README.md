# Data

Folder containing data files used to train the transformer.

## Format

The data file has to be a `csv` file with at least the following columns:

- a sentence or a full text
- a label (either 0/1 for binary classification or a survival time for regression tasks)

## Example

`french_tweets_short.csv` is a dataset example that we use while we are waiting for the real dataset.

You can also load the complete `french_tweets.csv` dataset (1M+ tweets) on [this Kaggle challenge](https://www.kaggle.com/hbaflast/french-twitter-sentiment-analysis/version/1).