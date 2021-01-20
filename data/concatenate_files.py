import os 

path_data = "/data/isilon/centraleNLP"
file_name_concatenated = "concatenate.txt"
list_files = os.listdir(path_data)

file_name_wanted = ["dcd.txt", "texteSimbad*"]

with open(os.path.join(path_data, file_name_concatenated), 'w') as outfile:
    for file_name in list_files:
        if "dcd.txt" in file_name or  "texteSimbad" in file_name:
            with open(os.path.join(path_data, file_name)) as infile:
                outfile.write(infile.read())



