python src/correction.py -d french_tweets_short.csv
python src/extract_unknown_words.py -d french_tweets_short.csv
python src/trainin.py -d french_tweets_short.csv
python src/hyperoptimization.py -e 2 -n 2 -d french_tweets_short.csv