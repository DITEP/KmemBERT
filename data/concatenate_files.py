import os

#test
path_data= "/data/isilon/centraleNLP"
file_name_concatenated = "concatenate.txt"
list_files = os.listdir(path_data)

file_name_wanted = ["dcd.txt", "texteSimbad*"]
header_is_written = False
with open(os.path.join(path_data, file_name_concatenated), 'w') as outfile:
    for file_name in list_files:
        if "dcd.txt" in file_name or  "texteSimbad" in file_name or "p2012.txt" in file_name:
            print(file_name)
            with open(os.path.join(path_data, file_name)) as infile:
                if header_is_written:
                    _ = infile.readline()
                else:
                    header_is_written = True
                    
                outfile.write(infile.read())      
