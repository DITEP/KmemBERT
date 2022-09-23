'''
Author : Théo Di Piazza (CLB)
Year : 2022

    Ce script permet de lire les données (xlsx), de nettoyer filtrer la data.
    Script to read all data (.xlsx), to clean and filter data.
    Note that this is not the only pre-processing phase.
    This is the first one, to make the files lighter. 

    Please change variables according to your need(s).

    Command line to execute: python testT2_data_read_clean.py 
'''

import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from utils import remove_duplicates, filtration_year, merge_patients, count_words, printc, time_to_event
from datetime import datetime
os.system('color')

printc("    Read all datas and clean it...", "WARNING")
##############################################################
# Patients: Lecture des jeux de donnees + suppression doublons
printc("    PATIENTS data opening... (1/2)", "WARNING")
patients_non_etudes = pd.read_excel("patients_not_essais_cliniques.xlsx", index_col=0)
patients_non_etudes = patients_non_etudes.drop(columns=['SEXE'])
patients_non_etudes = remove_duplicates(patients_non_etudes)

patients_etudes = pd.read_excel("patients.xlsx", index_col=0)
patients_etudes = patients_etudes.drop(columns=['ID_ETU', 'D_ENTREE_ETUDE', 'DMAJ_ETUDE'])
patients_etudes.rename(columns = {'DMAJ_PATIENT':'DMAJ'}, inplace = True)
patients_etudes = remove_duplicates(patients_etudes)

patients = merge_patients(patients_etudes, patients_non_etudes, "IPPR")
del patients_etudes, patients_non_etudes
printc("    PATIENTS data opened... (1/3)", "SUCCESS")

##############################################################
# Documents: Lecture des jeux de donnees + suppression des doublons
printc("    DOCUMENTS data opening... (2/2)", "WARNING")
dhe1 = pd.read_excel("documents_hors_essai_clinique_1.xlsx", index_col=0)
dhe1 = remove_duplicates(dhe1)
dhe2 = pd.read_excel("documents_hors_essai_clinique_2.xlsx", index_col=0)
dhe2 = remove_duplicates(dhe2)
dhe3 = pd.read_excel("documents_hors_essai_clinique_3.xlsx", index_col=0)
dhe3 = remove_duplicates(dhe3)

dhe = pd.concat([dhe1, dhe2, dhe3])
del dhe1, dhe2, dhe3
dhe = remove_duplicates(dhe)

# Read df with CR consultation
de1 = pd.read_excel("documents_1.xlsx", index_col=0)
de1 = remove_duplicates(de1)
de2 = pd.read_excel("documents_2.xlsx", index_col=0)
de2 = remove_duplicates(de2)
de3 = pd.read_excel("documents_3.xlsx", index_col=0)
de3 = remove_duplicates(de3)
de4 = pd.read_excel("documents_4.xlsx", index_col=0)
de4 = remove_duplicates(de4)
de5 = pd.read_excel("documents_5.xlsx", index_col=0)
de5 = remove_duplicates(de5)
de6 = pd.read_excel("documents_6.xlsx", index_col=0)
de6 = remove_duplicates(de6)
de7 = pd.read_excel("documents_7.xlsx", index_col=0)
de7 = remove_duplicates(de7)
de8 = pd.read_excel("documents_8.xlsx", index_col=0)
de8 = remove_duplicates(de8)
de9 = pd.read_excel("documents_9.xlsx", index_col=0)
de9 = remove_duplicates(de9)

de = pd.concat([de1, de2, de3, de4, de5, de6, de7, de8, de9])
del de1, de2, de3, de4, de5, de6, de7, de8, de9
de = remove_duplicates(de)
printc("    DOCUMENTS data opened... (2/3)", "SUCCESS")

######################################################
# Filter on a year
year_filter = 2000
# For Documents then patients
dhe = filtration_year(dhe, "DCREA", year_filter)
de = filtration_year(de, "DCREA", year_filter)
patients = filtration_year(patients, "DCREA", year_filter)

######################################################
# Concatenate 2 dataframes of documents, then filter on a year
doc_df = pd.concat([de, dhe])
del de, dhe
doc_df = remove_duplicates(doc_df)
all_patient = patients['IPPR']
doc_df = doc_df[doc_df['IPPR'].isin(all_patient)]

######################################################
# Filter on documents too long or too short
# Number of letters per document and number of words
doc_df['len_text'] = doc_df['text'].apply(len)
doc_df['nb_words'] = doc_df.apply(count_words, axis=1)
min_letters, max_words = 250, 350
doc_df = doc_df[(doc_df.len_text>=min_letters) & (doc_df.nb_words<=max_words)]

# Compute TIME SURVIVAL or TIME TO CENSOR for each patient
patients['TIME_SURVIVAL'] = patients.apply(time_to_event, axis=1)
# Add a FLAG column to know if the patient is censored or not
patients['FLAG_DECES'] = np.where(patients.DECES.isnull(), 0, 1)
# Remove censored patients that stopped to soon (before median of overall time to event)
quant1_surv, median_surv, quant3_surv = patients.TIME_SURVIVAL.quantile([0.25, 0.5, 0.75])
patients = patients[(patients.TIME_SURVIVAL>=median_surv) | (patients.FLAG_DECES==1)]
# Get documents with conserved patients
all_patient = patients.IPPR
doc_df = doc_df[doc_df['IPPR'].isin(all_patient)]

###########################################################################
# Conservation uniquement des CRC : Compte-Rendu Consultation
doc_df_CRC = doc_df[doc_df.Source == "Compte-rendu de consultation"]
# Remove duplicate IPPR (same IPPR but different person)
patients = patients[~patients['IPPR'].duplicated(keep=False)]
# Merge documents and patients dataframe
result = pd.merge(doc_df_CRC, patients[['IPPR', 'DECES', 'FLAG_DECES']], on="IPPR")

# Save the file
file_name = "doc_toTest\\doc_df_CRC_" + datetime.today().strftime('%Y-%m-%d') + ".csv"
result.to_csv(file_name, index=False)

printc("    Data read and concatenated. (3/3)", "SUCCESS")
###########################################################################
# END
"""
###########################################################################
# Pour conserver d autres types de CR
# Suppression des documents non intéressants (arbitraire)
source_to_remove = ['Ordonnance', 'Chiomiothérapie', 'Biologie']
doc_df = doc_df[~doc_df['Source'].isin(source_to_remove)]

# Recuperation des types de documents les plus frequents et leur pourcentage
most_frequent_source = doc_df['Source'].value_counts()[:150]#.index.tolist()
most_frequent_source_ind = doc_df['Source'].value_counts()[:150].index.tolist()
pourc = 100*most_frequent_source/sum(most_frequent_source)
pourc = ['%.2f' % elem for elem in pourc]

# Deuxieme filtre sur les CR, leurs noms, count et %
subsource_to_keep = ["CR", "Compte-rendu", "Imagerie"]
list1 = [item for item in most_frequent_source_ind if any(y in item for y in subsource_to_keep)]
list2 = [most_frequent_source[i] for i in range(len(most_frequent_source)) if any(y in most_frequent_source_ind[i] for y in subsource_to_keep)]
list3 = [pourc[i] for i in range(len(pourc)) if any(y in most_frequent_source_ind[i] for y in subsource_to_keep)]
list4 = np.cumsum([float(x) for x in list3]); list4 = ['%.2f' % elem for elem in list4]
# Selection des meilleurs types de document
list2 = [list1[i] for i in range(len(list1)) if float(list3[i])>=0.5]
list2.remove("CR de consultation d'anesthésie")
list2.remove("CR d'Anesthésie")
# On filtre sur ces types de documents
doc_df_CR = doc_df[doc_df.Source.isin(list2)]
doc_df_CR.to_csv("doc_df_CR.csv")
"""


