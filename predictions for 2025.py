import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from funtions import prepare_data
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

'''
The goal to this proyect is to try to predict the results of the 2024/2025 season using
the match data from the other seasons.

We will use the data from the 2024/2025 season to test how accurate the model is.
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
model_ovo.fit(X_train, y_train)

y_pred_ovo = model_ovo.predict(X_test)


## Evaluation of the model
print("One-vs-One (OvO) Strategy:")
labels = ['Home wins', 'Draw', 'Away Wins']
width = 15  # ancho de cada columna para alineación

# Calcular métricas
precision = precision_score(y_test, y_pred_ovo, average=None)
recall = recall_score(y_test, y_pred_ovo, average=None)
f1 = f1_score(y_test, y_pred_ovo, average=None)

# Construir filas alineadas
header = "".join(f"{label:<{width}}" for label in labels)
precision_row = "".join(f"{val:<{width}.4f}" for val in precision)
recall_row = "".join(f"{val:<{width}.4f}" for val in recall)
f1_row = "".join(f"{val:<{width}.4f}" for val in f1)

print("Model Evaluation Metrics (H/D/A):\n")
print(f"{'Results':<10}{header}")
print(f"{'Precision':<10}{precision_row}")
print(f"{'Recall':<10}{recall_row}")
print(f"{'F1-score':<10}{f1_row}")



## Features importances:
variables = X_train_scaled.columns
coefs = np.array([clf.coef_[0] for clf in model_ovo.estimators_])
feature_importance = np.mean(np.abs(coefs), axis=0)


# Graph
plt.figure(figsize=(10, 8))
plt.barh(variables, feature_importance)
plt.title("Feature Importance (OvO Logistic Regression)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()



### CREATION OF CLASSIFICATION TABLES

# Vamos a crear una tabla de clasificación:

teams = pd.unique(data_2025[['HomeTeam', 'AwayTeam']].values.ravel('K'))

predict_table = pd.DataFrame(0, index=teams, columns=['Points', 'Played', 'W', 'D', 'L'])

# Convertimos los códigos numéricos a letras (H, D, A)
result_mapping = dict(enumerate(data_until_2024['FTR'].astype('category').cat.categories))
pred_results = pd.Series(y_pred_ovo).map(result_mapping)   


# Recorrer todos los partidos de 2025 y sumar puntos
for index, row in data_2025.iterrows():
    home = row['HomeTeam']
    away = row['AwayTeam']
    pred = pred_results.iloc[index]  # Predicción para ese partido

    # Partidos jugados
    predict_table.loc[home, 'Played'] += 1
    predict_table.loc[away, 'Played'] += 1

    # Si gana el local
    if pred == 'H':
        predict_table.loc[home, 'Points'] += 3
        predict_table.loc[home, 'W'] += 1
        predict_table.loc[away, 'L'] += 1

    # Si hay empate
    elif pred == 'D':
        predict_table.loc[home, 'Points'] += 1
        predict_table.loc[away, 'Points'] += 1
        predict_table.loc[home, 'D'] += 1
        predict_table.loc[away, 'D'] += 1
    
    # Si gana el visitante
    else:  # 'A'
        predict_table.loc[away, 'Points'] += 3
        predict_table.loc[away, 'W'] += 1
        predict_table.loc[home, 'L'] += 1


predict_table = predict_table.sort_values(by=['Points', 'W'], ascending=False)

print('\n--- PREDICTION WITH THE MODEL ---')
print(predict_table)


# Creamos la tabla de clasificación real de 2025 con los datos del test:

classification_table = pd.DataFrame(0, index=teams, columns=['Points', 'Played', 'W', 'D', 'L'])
results = data_2025['FTR']

for i in range(380):
    home = data_2025.loc[i, 'HomeTeam']
    away = data_2025.loc[i, 'AwayTeam']
    r = results[i]

    classification_table.loc[home, 'Played'] += 1
    classification_table.loc[away, 'Played'] += 1

    if r == 'H':
        classification_table.loc[home, 'Points'] += 3
        classification_table.loc[home, 'W'] += 1
        classification_table.loc[away, 'L'] += 1

    elif r == 'D':
        classification_table.loc[home, 'Points'] += 1
        classification_table.loc[away, 'Points'] += 1
        classification_table.loc[home, 'D'] += 1
        classification_table.loc[away, 'D'] += 1

    else:  # 'A'
        classification_table.loc[away, 'Points'] += 3
        classification_table.loc[home, 'L'] += 1
        classification_table.loc[away, 'W'] += 1


classification_table = classification_table.sort_values(by=['Points', 'W'], ascending=False)

print('\n--- CLASSIFICATION SEASON 2024/2025 ---')
print(classification_table)


## Tabla de errores:

differences_table = predict_table - classification_table
differences_table = differences_table.reindex(index=predict_table.index, columns=predict_table.columns)
differences_table = differences_table.drop(['Played'], axis=1)


print('\n--- DIFFERENCE BETWEEN PREDICTION AND REALITY ---')
print(differences_table)

