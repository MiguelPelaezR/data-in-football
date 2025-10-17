import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

'''
The goal to this proyect is to try to predict the results of 2025 and 2026 using
the match data from the other seasons.

We will use the 2025 to test how accurate the model is.
'''

### We will start with the predictions for 2025:

# We store the different datasets:
data_until_2024 = pd.read_csv('datasets/Dataset from 2019 to 2024.csv',sep=';')
data_2025 = pd.read_csv('datasets/LaLiga_24_25_transform.csv', sep=';')


# Lets divide the data frame and categorize the data:
# For training the OvO model:
numeric_columns = [
    'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 
    'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 
    'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'PSH', 'PSD', 'PSA', 
    'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA'
]
match_data_train = data_until_2024[numeric_columns]
htr_train = data_until_2024['HTR'].astype('category').cat.codes
home_teams = data_until_2024['HomeTeam'].astype('category').cat.codes
away_teams = data_until_2024['AwayTeam'].astype('category').cat.codes



X_train = pd.concat([home_teams,away_teams, htr_train, match_data_train], axis=1)
std_scaler = StandardScaler()

X_train.columns = X_train.columns.astype(str)
X_train = std_scaler.fit_transform(X_train)

y_train = data_until_2024['FTR'].astype('category').cat.codes





# To test it
match_data_test = data_2025[numeric_columns]
htr_test = data_2025['HTR'].astype('category').cat.codes
home_teams = data_2025['HomeTeam'].astype('category').cat.codes
away_teams = data_2025['AwayTeam'].astype('category').cat.codes

X_test = pd.concat([home_teams,away_teams, htr_test, match_data_test], axis=1)

X_test.columns = X_test.columns.astype(str)
X_test = std_scaler.fit_transform(X_test)

y_test = data_2025['FTR'].astype('category').cat.codes



## Model OvO:
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)

y_pred_ovo = model_ovo.predict(X_test)

print("One-vs-One (OvO) Strategy:")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")


## Model OvA
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)

y_pred_ova = model_ova.predict(X_test)

print("One-vs-All (OvA) Strategy:")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")









data_until_2025 = pd.read_csv('datasets/Dataset from 2019 to 2025.csv')







