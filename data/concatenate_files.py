# This Python file uses the following encoding: utf-8
import os
import pandas as pd

path_data= "/data/isilon/centraleNLP"
file_name_concatenated = "concatenate.txt"
file_name_wanted = ["dcd.txt", "texteSimbad*"]

header = ["Noigr","clef","Date deces","Date cr","Code nature","Nature doct","Sce","Contexte","Texte"]

list_files = os.listdir(path_data)
file_path_concatenated = os.path.join(path_data, file_name_concatenated)
with open(file_path_concatenated, "w") as outfile:
    for file_name in list_files:
        if "dcd.txt" in file_name or  "texteSimbad" in file_name or "p2012.txt" in file_name:
            print(file_name)        
            with open(os.path.join(path_data, file_name)) as infile:
                header = infile.readline()
                if len(header.split("£")) == 9:
                    df = pd.read_csv(os.path.join(path_data, file_name), sep="\\£")
                elif len(header.split("|")) == 9:
                    df = pd.read_csv(os.path.join(path_data, file_name), sep="\\|")
                else:
                    continue

                df.to_csv(file_path_concatenated, header=header, index=None, sep="£", mode="a") 
                header = None