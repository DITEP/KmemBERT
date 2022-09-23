Author : Théo Di Piazza (CLB)
Year : 2022

###############################################################################
These scripts permit to read raw data, clean it, prepare it then pre-process it 
to test kmembert-T2 from CLB datas.

Please, make sure you have all the necessary data in your fodler.
Necessary datasets are indicated at the end of this .txt file.
###############################################################################

###############################################################################
3 scripts (.py) :

- testT2_data_read_clean.py : to open raw datasets and concatenate to make the files
lighter. Takes around 20mn to run.
- testT2_data_prepare.py : to extract 'n' patients of the dataset created by the
precedent file.
- testT2_data_preprocessing.py : some preprocessing and cleaning steps to put data into
the model.

Please, run these 3 scripts in the above order.
###############################################################################

Files needed:
document_1.xlsx, ... document_9.xlsx
documents_hors_essai_clinique_1.xlsx, ..., documents_hors_essai_clinique_3.xlsx
patients, patients_not_essai_clinique.xlsx
###############################################################################

End, thank you.
###############################################################################
