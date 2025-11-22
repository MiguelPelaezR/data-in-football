import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


'''
We will evaluate how accurate XGBoost is 
'''

data_until_2024 = pd.read_csv('datasets/Dataset from 2019 to 2024.csv',sep=';')
data_2025 = pd.read_csv('datasets/LaLiga_24_25_transform.csv', sep=';')

# We store the different datasets:
data_until_2024 = pd.read_csv('datasets/Dataset from 2019 to 2024.csv',sep=';')
data_2025 = pd.read_csv('datasets/LaLiga_24_25_transform.csv', sep=';')



# Lets divide the data frame and categorize the data:
# For training the OvO model:
numeric_columns = [
    'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 
    'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
]

bets_columns = [
    'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'PSH', 'PSD', 'PSA', 
    'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA' 
]

match_data_train = data_until_2024[numeric_columns]
bets_data_train = data_until_2024[bets_columns]
htr_train = data_until_2024['HTR'].astype('category').cat.codes
home_teams = data_until_2024['HomeTeam'].astype('category').cat.codes
away_teams = data_until_2024['AwayTeam'].astype('category').cat.codes



X_train = pd.concat([home_teams,away_teams, htr_train, match_data_train, bets_data_train], axis=1)
X_train_without_bets = pd.concat([home_teams,away_teams, htr_train, match_data_train], axis=1)

std_scaler = StandardScaler()

X_train.columns = X_train.columns.astype(str)
X_train = std_scaler.fit_transform(X_train)

y_train = data_until_2024['FTR'].astype('category').cat.codes



# To test it
match_data_test = data_2025[numeric_columns]
bets_data_test = data_2025[bets_columns]
htr_test = data_2025['HTR'].astype('category').cat.codes
home_teams = data_2025['HomeTeam'].astype('category').cat.codes
away_teams = data_2025['AwayTeam'].astype('category').cat.codes

X_test = pd.concat([home_teams,away_teams, htr_test, match_data_test, bets_data_test], axis=1)
X_test_without_bets = pd.concat([home_teams,away_teams, htr_test, match_data_test], axis=1)

X_test.columns = X_test.columns.astype(str)
X_test = std_scaler.fit_transform(X_test)

y_test = data_2025['FTR'].astype('category').cat.codes

## initialize the model xgboost:

xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,objective='multi:softmax', num_class=3)

### WITH BETS:

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

#### Evaluation:

print(f"XGBoost Evaluation:")
print(f" Accuracy: {np.round(100*accuracy_score(y_test, y_pred_xgb),2)}%")
print(f" F1 macro: {f1_score(y_test, y_pred_xgb, average='macro'):.4f}")
print("\n Confusion matrix:")
print(confusion_matrix(y_test, y_pred_xgb))



### WIHTOUT BETS:
xgb.fit(X_train_without_bets, y_train)
y_pred_no_bets = xgb.predict(X_test_without_bets)

#### Evaluation:

print(f"\n XGBoost Evaluation WITH OUT BETS:")
print(f" Accuracy: {np.round(100*accuracy_score(y_test, y_pred_no_bets),2)}%")
print(f" F1 macro: {f1_score(y_test, y_pred_no_bets, average='macro'):.4f}")
print("\n Confusion matrix:")
print(confusion_matrix(y_test, y_pred_no_bets))
'''
The model works better without bets rather than with the bets variables
'''


