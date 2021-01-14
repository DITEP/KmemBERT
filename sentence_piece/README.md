# SentencePiece Training

Create a `.txt` file containg a text. Use the following command line to train SentencePiece on this test file.

## Command line example

`spm_train --input=voc.txt --model_prefix=test --vocab_size=80 --character_coverage=1.0 --model_type=bpe`

> Note: We usually use `BPE` as a model type.
> Warning: you have to choose carefully the parameters. Please refer to the [original repository](https://github.com/google/sentencepiece).