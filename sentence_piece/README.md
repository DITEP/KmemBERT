# SentencePiece Training

Create a `.txt` file containg a text. Use the following command line (choose your own parameters) to train SentencePiece on this test file.

`spm_train --input=voc.txt --model_prefix=test --vocab_size=80 --character_coverage=1.0 --model_type=bpe`