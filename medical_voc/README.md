# Medical vocabulary

This folder provides multiple files containing medical vocabulary.

## Model vocabulary extension

Camembert vocabulary can be expended with the `.json` files in here. These files contain frequent words from the full dataset that are not contained in the tokenizer.

Their names indicates their size, and the `p_` prefix means the file has been preprocessed (same words with an 's' and misspelled words been removed).

## Correction

`fr-100k.txt` is a long file containing medical words (among others). It can be used for spell checking.