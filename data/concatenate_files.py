# -*- coding: utf-8 -*-
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
                if len(header.split('£')) == 9:
                    sep = '£'
                elif len(header.split('|')) == 9:
                    sep = '|'
                else:
                    continue
                
                n_errors = 0
                for i, line in enumerate(infile):
                    row = line.split(sep)
                    if len(row) == 9:
                        outfile.write('|'.join(row))
                    else:
                        n_errors+=1
                        
                print("{} errors out of {} lines".format(n_errors, i))