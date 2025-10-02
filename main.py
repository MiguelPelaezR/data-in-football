import pandas as pd
import numpy as np



df = pd.read_csv('datasets/LaLiga_24_25.csv')

titles = list(df.columns)

'''print(sorted(titles))
print(df['FTR'].value_counts()['H'])
print(df['FTR'].value_counts()['D'])
print(df['FTR'].value_counts()['A'])'''



print(list(df.isna().sum()))



incompleto = [item for item in titles if df[item].isna().any()]

print(incompleto)
