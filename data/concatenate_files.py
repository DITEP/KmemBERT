import os
import pandas as pd

path_data= "/data/isilon/centraleNLP"
file_name_concatenated = "concatenate.txt"
file_name_wanted = ["dcd.txt", "texteSimbad*"]
seps = ["\xc2\xa3", "|", "@"]
new_sep = "\xc2\xa3"
new_line = "\n"

columns = ["Noigr","clef","Date deces","Date cr","Code nature","Nature doct","Sce","Contexte","Texte"]

list_files = os.listdir(path_data)
file_path_concatenated = os.path.join(path_data, file_name_concatenated)
with open(file_path_concatenated, "w") as outfile:
    for file_name in list_files:
        if "dcd.txt" in file_name or  "texteSimbad" in file_name or "p2012.txt" in file_name:
            print(file_name)        
            with open(os.path.join(path_data, file_name)) as infile:
                header = infile.readline()
                for sep in seps:
                    if len(header.split(sep)) == 9:
                        break

                if len(header.split(sep)) != 9:
                    print(repr(header))
                    continue

                if columns:
                    outfile.write(new_sep.join(columns) + new_line)
                    columns = None

                n_errors = 0
                for i, line in enumerate(infile):
                    row = line.split(sep)
                    if len(row) == 9 and (sep==new_sep or len(line.split(new_sep)) == 1):
                        line = new_sep.join(row)
                        if line[-1] != new_line:
                            line += new_line

                        outfile.write(line)
                    else:
                        n_errors+=1
                        
                print("{} errors out of {} lines".format(n_errors, i))