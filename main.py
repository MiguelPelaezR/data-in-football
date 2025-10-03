import pandas as pd
import numpy as np



def LimpiarDF(df):
    incompleto = [item for item in titles if df[item].isna().any()]
    return df.drop(incompleto)








df = pd.read_csv('datasets/LaLiga_24_25.csv')

titles = list(df.columns)

'''print(sorted(titles))
print(df['FTR'].value_counts()['H'])
print(df['FTR'].value_counts()['D'])
print(df['FTR'].value_counts()['A'])'''



print(list(df.isna().sum()))

print('\n\n')

incompleto = [item for item in titles if df[item].isna().any()]
cantidad_incompleto = (df[incompleto].isna().sum())



print(cantidad_incompleto)


