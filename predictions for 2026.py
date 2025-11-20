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
The goal to this proyect is to try to predict the results of 2025 using
the match data from the other seasons.

We will use the 2025 to test how accurate the model is.
'''

### We will start with the predictions for 2025:

# We store the different datasets:
data_until_2024 = pd.read_csv('datasets/Dataset from 2019 to 2024.csv',sep=';')
data_2025 = pd.read_csv('datasets/LaLiga_24_25_transform.csv', sep=';')

print(data_2025.columns)

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




# Vamos a crear una tabla de clasificación:

teams = pd.unique(data_2025[['HomeTeam', 'AwayTeam']].values.ravel('K'))

predict_table = pd.DataFrame(0, index=teams, columns=['Points', 'Played', 'W', 'D', 'L'])

# Convertimos los códigos numéricos a letras (H, D, A)
result_mapping = dict(enumerate(data_until_2024['FTR'].astype('category').cat.categories))
pred_results = pd.Series(y_pred_ova).map(result_mapping)   


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

print(classification_table)


## Tabla de errores: