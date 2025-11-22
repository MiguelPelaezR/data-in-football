import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from funtions import prepare_data

import warnings
warnings.filterwarnings('ignore')

'''
We will evaluate how accurate the Decission Tree and SVM models are  
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

# Split data into training and test sets for training the Regression model:
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

# Inicializate the Decision tree model:

dec_tree = DecisionTreeClassifier(max_depth=6)

### WITH BETS:
dec_tree.fit(X_train_scaled, y_train)

y_pred_dt = dec_tree.predict(X_test_scaled)

print("~~~~ Decission Tree Strategy: ~~~~")

print('\n--- ACCURACY WITH BETS ---')
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_dt),2)}%")
print(f"F1 macro: {f1_score(y_test, y_pred_dt, average='macro'):.4f}")
print("\n Confusion matrix:")
print(confusion_matrix(y_test, y_pred_dt))


### WITH OUT BETS:
dec_tree.fit(X_train_no_bets_scaled, y_train)

y_pred_dt_no_bets = dec_tree.predict(X_test_no_bets_scaled)

print('\n--- ACCURACY WITH OUT BETS ---')
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_dt_no_bets),2)}%")
print(f"F1 macro: {f1_score(y_test, y_pred_dt_no_bets, average='macro'):.4f}")
print("\n Confusion matrix:")
print(confusion_matrix(y_test, y_pred_dt_no_bets))





# Inicializate the Support Vector Machine (SVM) model 
svm = LinearSVC(class_weight='balanced', loss="hinge")

svm.fit(X_train_scaled, y_train)

y_pred_svm = svm.predict(X_test_scaled)

print("\n\n~~~~SVM Strategy: ~~~~")

print('\n--- ACCURACY WITH BETS ---')
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_svm),2)}%")
print(f"F1 macro: {f1_score(y_test, y_pred_svm, average='macro'):.4f}")
print("\n Confusion matrix:")
print(confusion_matrix(y_test, y_pred_svm))

### WITH OUT BETS:
svm.fit(X_train_no_bets_scaled, y_train)

y_pred_svm_no_bets = dec_tree.predict(X_test_no_bets_scaled)

print('\n--- ACCURACY WITH OUT BETS ---')
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_svm_no_bets),2)}%")
print(f"F1 macro: {f1_score(y_test, y_pred_svm_no_bets, average='macro'):.4f}")
print("\n Confusion matrix:")
print(confusion_matrix(y_test, y_pred_svm_no_bets))
