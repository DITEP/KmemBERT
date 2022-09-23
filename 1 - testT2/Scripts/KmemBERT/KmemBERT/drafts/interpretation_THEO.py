'''
    From a given index of Consultation, plot and save Words Importance
'''

# Import usefull libraries
if(True):
    import numpy as np
    import pandas as pd
    import argparse
    import shap
    from torch.serialization import save
    from transformers_interpret import SequenceClassificationExplainer
    import matplotlib.pyplot as plt

    from kmembert.models import HealthBERT
    from kmembert.utils import create_session

# Import argparse
if(True):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="data/ehr/test.csv", 
        help="data path to access to the testing file")
    parser.add_argument("-p", "--path_dataset", type=str, default="data/ehr/test.csv", 
        help="data path to access to the testing file")
    parser.add_argument("-r", "--resume", type=str, default="kmembert-base", 
        help="result folder in with the saved checkpoint will be reused")
    parser.add_argument("-nr", "--nrows", type=int, default=10, 
        help="maximum number of samples for testing")
    parser.add_argument("-f", "--folder_to_save", type=str, default="graphs", 
        help="folder to save the figures")
    parser.add_argument("-ng", "--noigr", type=int, default=2, 
        help="The Noigr of a patient")
    args = parser.parse_args("")

# Load Model
if(True):
    _, _, device, config = create_session(args)
    model = HealthBERT(device, config)

# Select Patient and a text to classify
if(True):
    noigr = 17186226
    file_to_classify = pd.read_csv(config.data_folder)
    texts_to_classify = file_to_classify.loc[file_to_classify.Noigr == noigr].Texte.values
    texts_to_classify = texts_to_classify[:1]

# Print and save the fig
if(True):
    word_attributes = []
    for ehr in texts_to_classify:
        cls_explainer = SequenceClassificationExplainer(
            model.camembert,
            model.tokenizer)
        word_attributions = cls_explainer(ehr)
        word_attributes+= [dict(word_attributions)]

    result = {}
    n_documents = len(word_attributes)
    for d in word_attributes:
        for k in d.keys():
            result[k] = result.get(k, 0) + d[k]/n_documents

    show_n = 10
    final_dict = dict(sorted(result.items(), key=lambda item: abs(item[1]))[:show_n])

    # Plot the results for the word attributes (green for positive contribution and red for negative contribution)
    x = list(final_dict.keys())
    y = list(final_dict.values())
    colors = ['deeppink']*show_n 
    for i in range(len(y)):
        if y[i]>0:
            colors[i] = 'dodgerblue'
    plt.figure()
    plt.barh(x, y, color=colors)
    plt.title(f'Word Attributions for all EHRs of the patient {noigr}')
    plt.savefig(f'graphs/interpretation/test_{noigr}.png')    
    plt.close()

# Display Text with colors
if(False):
    # Modification du dictionnaire pour afficher les résultats
    # Set every values to 0 to the dict with full sentence
    for k, v in result.items():
        result[k] = 0
    # Add values to words to be plotted
    for k, v in final_dict.items():
        if k in result.keys():
            result[k] = v

    # Mise en forme pour affichage
    txt = list(result.keys())
    txt_data = (list(map(( lambda x: x+' '), txt)),)

    val = list(result.values())
    txt_values = np.array([val])

    # Création de l'objet shap et affichage
    test = shap._explanation.Explanation(values=txt_values)
    test.data = txt_data
    test.base_values = np.array([-2.16531396])
    shap.plots.text(test[0])


############## END



