import pandas as pd
import numpy as np
import sys
sys.path.append("../../src/")
from utils import get_label



def remover_rows(data):
    df = pd.read_csv(data)
    condition= np.array(list(df[["Date deces", "Date cr"]].apply(lambda x: get_label(*x), axis=1)))
    print(condition)
    new_df =  df[condition>0]
    print(new_df)
    return new_df




if __name__ == '__main__':
    new_train =remover_rows('train.csv') 
    new_test = remover_rows('test.csv')

    new_train.to_csv('New_train.csv')
    new_test.to_csv('New_test.csv')