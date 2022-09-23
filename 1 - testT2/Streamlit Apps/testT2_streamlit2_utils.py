'''
Autheur : Théo Di Piazza
Centre Léon Bérard, Lyon
2022
'''
import pandas as pd
from datetime import datetime
import sys
import os

bcolors = {
    'RESULTS': '\033[95m',
    'HEADER': '\033[94m',
    'SUCCESS': '\033[92m',
    'WARNING': '\033[93m',
    'FAIL': '\033[91m',
    'ENDC': '\033[0m',
    'INFO': '\033[1m',
    'UNDERLINE': '\033[4m'
}

def printc(log, color='HEADER'):
    """
    Prints logs with color according to the dict bcolors
    """
    print(f"{bcolors[color]}{log}{bcolors['ENDC']}")

def remove_duplicates(df: pd.DataFrame):
    """
        Remove duplicates from a dataframe and print number of elements removed
        Input: a dataframe
        Output: a dataframe with removed duplicates
    """
    #print("Number of elements in the initial dataset: ", len(df))
    res = df.drop_duplicates(keep="first")
    #print("Number of elements in the initial dataset: ", len(res))
    #print("Number of elements removed:", len(df) - len(res))
    #print("-------------------------------")
    return res

def filtration_year(df: pd.DataFrame, colname:str, year: int):
    """
        Remove elements (patients or documents) taken into account before a given year
        Input: dataframe, colname: name of the column, year: year of filtration (>=year)
        Output: dataframe filtered
    """
    #print("Number of elements before the filter:", len(df))

    res = df
    res['DCREA_year'] = res[colname].dt.year # Turn Time column to Int (year)
    res = res[res['DCREA_year'] >= year] # Filter

    #print("Number of elements after the filter:", len(res))
    return res

def merge_patients(df1: pd.DataFrame, df2: pd.DataFrame, colname: str = "IPPR"):
    """
        Extract ID of all patients, concatenate 2 dfs and remove duplicates
        Input: 2 dfs. dfs need to have same column
                colname: name of the column with patients ID
    """
    # Extract ID 
    list1 = df1[colname].tolist()
    list2 = df2[colname].tolist()
    #print("Number of elements extracted from 2 dataframes:", len(list1)+len(list2))
    all_patient = list(set(list1+list2)) # Concatenate all

    # Concatenate, filter by patients in and remove duplicates
    res = pd.concat([df1, df2])
    res = res[res[colname].isin(all_patient)]
    res = remove_duplicates(res)
    #print("Number of elements after concatenation:", len(res))

    return res

def count_words(row):
    val = len(row.text.split())
    return val

def time_to_event(row):
    if pd.isnull(row['DECES']):
        val = (row.DMAJ - row.DCREA).days
    else:
        val = (row.DECES - row.DCREA).days
    return val

def strDate_to_days(row, date_format = "%Y-%m-%d"):
    '''
        When there's a FLAG_DECES column in the df
    '''
    if(row['FLAG_DECES']==1):
        a = datetime.strptime(row['Date deces'], date_format)
        b = datetime.strptime(row['Date creation'], date_format)
        val = (a-b).days
    else:
        a = datetime.strptime(row['Date derniere maj'], date_format)
        b = datetime.strptime(row['Date creation'], date_format)
        val = (a-b).days
    return val

def strDate_to_days_bis(row, date_format = "%Y-%m-%d"):
    '''
        When there's no FLAG_DECES column in the df
    '''
    a = datetime.strptime(row['Date deces'], date_format)
    b = datetime.strptime(row['Date creation'], date_format)
    val = (a-b).days
    return val

def add_dateCreation(df):
    '''
        For streamlit app visualisation, deal with dates format and date creation
        from patients dataframe
    '''
    # Deal with dates
    # Date deces
    df['Date deces'] = df['Date deces'].apply(lambda x: str(x))
    df['Date deces'] = df['Date deces'].apply(lambda x: '-'.join([x[:4], x[4:6], x[6:]]))
    # Date cr
    df['Date cr'] = df['Date cr'].apply(lambda x: str(x))
    df['Date cr'] = df['Date cr'].apply(lambda x: '-'.join([x[:4], x[4:6], x[6:]]))

    # Open dataframe about patients
    patients_non_etudes = pd.read_excel("patients_not_essais_cliniques.xlsx", index_col=0)
    patients_non_etudes = patients_non_etudes.drop(columns=['SEXE'])
    patients_non_etudes = remove_duplicates(patients_non_etudes)
    patients_etudes = pd.read_excel("patients.xlsx", index_col=0)
    patients_etudes = patients_etudes.drop(columns=['ID_ETU', 'D_ENTREE_ETUDE', 'DMAJ_ETUDE'])
    patients_etudes.rename(columns = {'DMAJ_PATIENT':'DMAJ'}, inplace = True)
    patients_etudes = remove_duplicates(patients_etudes)
    patients = merge_patients(patients_etudes, patients_non_etudes, "IPPR")
    # Merge and select columns
    result = df.merge(patients, left_on='Noigr', right_on='IPPR')[['Noigr', 'DCREA', 'Date cr', 'Date deces', 'Texte']]
    result.rename(columns = {'DCREA':'Date creation'}, inplace = True)
    result['Date creation'] = result['Date creation'].apply(lambda x: str(x)[:10])
    del patients_etudes, patients_non_etudes, patients

    return result
