'''
    Théo Di Piazza, 2022, CLB
    Lyon

    Usefull functions for data manipulation, for French EDA implementation
'''

import random
import math
import spacy
import json

from nltk.corpus import wordnet

# Fonction pour supprimer des mots aléatoires
def remove_words(txt: str, pourcentage: float):
    '''
        From a given text 'txt', remove a 'pourcentage' of words in the text
    '''
    # Split text into list
    lst_txt = txt.split()

    # Number of words to remove
    nb_remove = math.floor(pourcentage*len(lst_txt))

    # List with index of elements to remove
    index_remove = random.sample(range(0, len(lst_txt)), nb_remove)

    # For loop on the text to remove words
    for i in range(len(lst_txt)):
        # If i is an index to remove
        if(i in index_remove):
            # Then, remove the word
            print(i)
            del lst_txt[i]

    return " ".join(lst_txt)

# Fonction v1 pour remplacer les synonymes
def replace_synonyms(txt: str, pourcentage: float):
    '''
    From a given text 'txt', replace a given 'pourcentage' of words
    '''
    # Language 
    lang = 'fra'

    # Split text into list
    lst_txt = txt.split()

    # Number of words to replace
    nb_remove = math.floor(pourcentage*len(lst_txt))

    # List with index of elements to replace
    index_remove = random.sample(range(0, len(lst_txt)), nb_remove)

    # For loop on the text to replace words
    for i in range(len(lst_txt)):
        # If i is an index to replace
        if(i in index_remove):
            # Get word to replace
            wrd = lst_txt[i]
            # Initialize list of synonyms
            synonyms = []
            # Make list of synonyms for a given word
            for ss in [n for synset in wordnet.synsets(wrd, lang=lang) for n in synset.lemma_names(lang)]:
                synonyms.append(ss)
            # Make synonyms as a list of str (words)
            synonyms = list(filter(("petit").__ne__, synonyms))

            # If some synonyms are found
            if(synonyms):
                # Select randomly a synonym
                synonym_selected = random.choice(synonyms)
                # Replace the word by a synonym
                lst_txt[i] = synonym_selected

    return " ".join(lst_txt)

def replace_synonyms_v2(txt: str, pourcentage: float=1):
    '''
    From a given text 'txt', replace a given 'pourcentage' of words
    '''
    # Language 
    lang = 'fra'

    # Load French Large Spacy model
    nlp = spacy.load('fr_core_news_lg')
    # Load list of french words
    from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop

    # Split text into list
    lst_txt = txt.split()

    # Number of words to replace
    nb_remove = math.floor(pourcentage*len(lst_txt))

    # List with index of elements to replace
    index_remove = random.sample(range(0, len(lst_txt)), nb_remove)

    # List of medical words, words to not remove
    medical_list = medical_voc_list()

    # For loop on the text to replace words
    for i in range(len(lst_txt)):
        # Get word to replace
        wrd = lst_txt[i].lower()
        # If i is an index to replace AND there is only alpha-numeric character in the word AND its not a stop-word
        if(i in index_remove and wrd.isalnum() and (not wrd in fr_stop) and (not wrd in medical_list)):
            # Initialize list of synonyms
            synonyms = []
            # Make list of synonyms for a given word
            for ss in [n for synset in wordnet.synsets(wrd, lang=lang) for n in synset.lemma_names(lang)]:
                # If there is only alpha-numeric character
                if(ss.isalnum()):
                    # Add the synonym to the list
                    synonyms.append(ss.lower())
                    # Make synonyms as a list of str (words), without the current word
                    synonyms = list(filter((wrd).__ne__, synonyms))

            # If at least 1 synonym is found
            if(synonyms):
                # Filter on 5 best synonyms
                wrd_nlp = nlp(wrd)
                # Get similarities for each synonym
                synonyms_similarities = list(map(lambda x: nlp(x).similarity(wrd_nlp), synonyms))
                # Make a dict from these similarities, such as {synonyms: similarities}
                syn_results = dict(zip(synonyms, synonyms_similarities))
                # Filter on synonyms whose similarity with current word is upper than 0.5
                syn_results = {k: v for k, v in syn_results.items() if v > 0.2}
                # Extract 5 synonyms with highest similarity
                syn_results = sorted(syn_results, key=syn_results.get, reverse=True)[:5]
                # If we still have at least one synonym in the list
                if(syn_results):
                    # Select randomly a synonym
                    synonym_selected = random.choice(syn_results)
                    # Replace the word by a synonym
                    lst_txt[i] = synonym_selected

    # Now, concatenate every words of the list to return
    return " ".join(lst_txt)



# Fonction qui retourne une liste des mots du vocabulaire médical
def medical_voc_list():
    '''
        Returns a list of medical words from large.json
    '''
    # List to save medical words
    medical_words = []
    # Length minimum to save words
    length_min = 2

    # Open vocabulary
    with open('vocabs/large.json', encoding='utf-8') as fh:
        data = json.load(fh)

    # For each word, save it if its length is superior to length_min
    for word in data:
        # If the length word is superior to length_min
        if(len(word[0])>length_min):
            # Add it to the list
            medical_words.append(word[0])

    return medical_words




