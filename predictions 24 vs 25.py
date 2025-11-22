import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from funtions import prepare_data

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


df_2024 = pd.read_csv('datasets/LaLiga_23_24_transform.csv', sep=';')
df_2025 = pd.read_csv('datasets/LaLiga_24_25_transform.csv', sep=';')


#Upon review, it was observed that all columns containing missing values are related to betting data; therefore, these variables will be excluded from the analysis.
numeric_columns = [
    'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 
    'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
]

bets_columns = [
    'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'PSH', 'PSD', 'PSA', 
    'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA' 
]

# Split data into training and test sets for training the OvO model:
X_train, X_train_without_bets = prepare_data(df_2024, numeric_columns, bets_columns)
X_test, X_test_without_bets = prepare_data(df_2025, numeric_columns, bets_columns)

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns.astype(str))
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns.astype(str))

X_train_no_bets_scaled = pd.DataFrame(scaler.fit_transform(X_train_without_bets),columns=X_train_without_bets.columns.astype(str))
X_test_no_bets_scaled = pd.DataFrame(scaler.transform(X_test_without_bets),columns=X_test_without_bets.columns.astype(str))

y_train = df_2024['FTR'].astype('category').cat.codes
y_test = df_2025['FTR'].astype('category').cat.codes

### With One vs All method:
model_ova = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model_ova.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_ova = model_ova.predict(X_test_scaled)

# Calculate accuracy with the bets variables
print("~~~~~~ Strategy with the bets variables ~~~~~~")

precision = precision_score(y_test, y_pred_ova, average=None)
recall = recall_score(y_test, y_pred_ova, average=None)
f1 = f1_score(y_test, y_pred_ova, average=None)

# Creation of a evaluation table
labels = ['Home wins', 'Draw', 'Away Wins']
width = 15  # how width the table is
header = "".join(f"{label:<{width}}" for label in labels)
precision_row = "".join(f"{val:<{width}.4f}" for val in precision)
recall_row = "".join(f"{val:<{width}.4f}" for val in recall)
f1_row = "".join(f"{val:<{width}.4f}" for val in f1)

print("Model Evaluation Metrics (H/D/A) with bets:\n")
print(f"{'Results':<10}{header}")
print(f"{'Precision':<10}{precision_row}")
print(f"{'Recall':<10}{recall_row}")
print(f"{'F1-score':<10}{f1_row}")

print(f'\nAccuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%')
print(confusion_matrix(y_test, y_pred_ova))

# Calculate accuracy with OUT the bets variables
ova_no_bets = model_ova.fit(X_train_no_bets_scaled, y_train)
y_pred_no_bets = model_ova.predict(X_test_no_bets_scaled)

print("~~~~~~ Strategy WITH OUT the bets variables ~~~~~~")

precision = precision_score(y_test, y_pred_no_bets, average=None)
recall = recall_score(y_test, y_pred_no_bets, average=None)
f1 = f1_score(y_test, y_pred_no_bets, average=None)

# Creation of a evaluation table
labels = ['Home wins', 'Draw', 'Away Wins']
width = 15  # how width the table is
header = "".join(f"{label:<{width}}" for label in labels)
precision_row = "".join(f"{val:<{width}.4f}" for val in precision)
recall_row = "".join(f"{val:<{width}.4f}" for val in recall)
f1_row = "".join(f"{val:<{width}.4f}" for val in f1)

print("Model Evaluation Metrics (H/D/A) WITH OUT bets variables:\n")
print(f"{'Results':<10}{header}")
print(f"{'Precision':<10}{precision_row}")
print(f"{'Recall':<10}{recall_row}")
print(f"{'F1-score':<10}{f1_row}")

print(f'\nAccuracy: {np.round(100*accuracy_score(y_test, y_pred_no_bets),2)}%')
print(confusion_matrix(y_test, y_pred_no_bets))





