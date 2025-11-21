import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from funtions import prepare_data
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

'''
We will evaluate how accurate teh models One vs One and One vs All are 
'''


# We store the different datasets:
data_until_2024 = pd.read_csv('datasets/Dataset from 2019 to 2024.csv',sep=';')
data_2025 = pd.read_csv('datasets/LaLiga_24_25_transform.csv', sep=';')



# Lets divide the data frame and categorize the data:

numeric_columns = [
    'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 
    'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
]

bets_columns = [
    'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'PSH', 'PSD', 'PSA', 
    'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA' 
]

# Split data into training and test sets for training the OvO model:
X_train, X_train_without_bets = prepare_data(data_until_2024, numeric_columns, bets_columns)
X_test, X_test_without_bets = prepare_data(data_2025, numeric_columns, bets_columns)

# Standarize the data:
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns.astype(str))
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns.astype(str))

X_train_no_bets_scaled = pd.DataFrame(scaler.fit_transform(X_train_without_bets),columns=X_train_without_bets.columns.astype(str))
X_test_no_bets_scaled = pd.DataFrame(scaler.transform(X_test_without_bets),columns=X_test_without_bets.columns.astype(str))

y_train = data_until_2024['FTR'].astype('category').cat.codes
y_test = data_2025['FTR'].astype('category').cat.codes

## initialize the model OvO:

model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))

### WITH BETS:

model_ovo.fit(X_train, y_train)

y_pred_ovo = model_ovo.predict(X_test)

print("One-vs-One (OvO) Strategy:")
print('--- ACCURACY WITH BETS ---')
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")
print(f" F1 macro: {f1_score(y_test, y_pred_ovo, average='macro'):.4f}")
print("\n Confusion matrix:")
print(confusion_matrix(y_test, y_pred_ovo))

### WITH OUT BETS:

model_ovo.fit(X_train_without_bets, y_train)

y_pred_ovo_no_bets = model_ovo.predict(X_test_without_bets)

print('\n--- ACCURACY WITH OUT BETS ---')
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo_no_bets),2)}%")
print(f"F1 macro: {f1_score(y_test, y_pred_ovo_no_bets, average='macro'):.4f}")
print("\n Confusion matrix:")
print(confusion_matrix(y_test, y_pred_ovo_no_bets))



## INICIALIZATE OvA MODEL:
model_ova = OneVsRestClassifier(LogisticRegression(max_iter=1000))

### WITH BETS:
model_ova.fit(X_train,y_train)

y_pred_ova = model_ova.predict(X_test)

print("One-vs-All (OvA) Strategy:")
print('--- ACCURACY WITH BETS ---')
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")
print(f" F1 macro: {f1_score(y_test, y_pred_ova, average='macro'):.4f}")
print("\n Confusion matrix:")
print(confusion_matrix(y_test, y_pred_ova))

### WITH OUT BETS:

model_ova.fit(X_train_without_bets, y_train)

y_pred_ova_no_bets = model_ova.predict(X_test_without_bets)

print('\n--- ACCURACY WITH OUT BETS ---')
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova_no_bets),2)}%")
print(f"F1 macro: {f1_score(y_test, y_pred_ova_no_bets, average='macro'):.4f}")
print("\n Confusion matrix:")
print(confusion_matrix(y_test, y_pred_ova_no_bets))

