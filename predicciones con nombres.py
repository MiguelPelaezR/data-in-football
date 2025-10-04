import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.multiclass import OneVsOneClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


df_2024 = pd.read_csv('datasets/LaLiga_23_24.csv')
df_2025 = pd.read_csv('datasets/LaLiga_24_25.csv')

#Upon review, it was observed that all columns containing missing values are related to betting data; therefore, these variables will be excluded from the analysis.
df_2024 = df_2024.dropna(axis=1)
df_2025 = df_2025.dropna(axis=1)


# Quiero que tenga en cuenta que equipo juega, hacemos estas variables categóricas
'''df_2024['HomeTeam'] = df_2024['HomeTeam'].astype('category')
df_2024['AwayTeam'] = df_2024['AwayTeam'].astype('category')
df_2024['FTR'] = df_2024['FTR'].astype('category')'''

# Lets divide the data frame in to some features:
match_data_2024 = df_2024[['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]
home_teams_2024 = df_2024['HomeTeam'].astype('category').cat.codes
away_teams_2024 = df_2024['AwayTeam'].astype('category').cat.codes

X_train = pd.concat([home_teams_2024,away_teams_2024, match_data_2024], axis=1)
y_train = df_2024['FTR'].astype('category').cat.codes


match_data_2025 = df_2025[['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]
home_teams_2025 = df_2025['HomeTeam'].astype('category').cat.codes
away_teams_2025 = df_2025['AwayTeam'].astype('category').cat.codes

X_test = pd.concat([home_teams_2025,away_teams_2025, match_data_2025], axis=1)
y_test = df_2025['FTR'].astype('category').cat.codes


X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)




### OVA


model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)

# Make predictions on the test set
y_pred_ova = model_ova.predict(X_test)

# Calculate accuracy
print("One-vs-All (OvA) Strategy:")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")

variables = ['HomeTeam', 'AwayTeam', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']




#  Contar victorias del Real Madrid
# Añadir las predicciones al dataframe de test
df_2025['pred_FTR_code'] = y_pred_ova

# Mapear los códigos a los resultados reales ('H', 'D', 'A')
df_2025['pred_FTR'] = df_2025['pred_FTR_code'].map(
    dict(enumerate(df_2025['FTR'].astype('category').cat.categories))
)


madrid_home_wins = df_2025[
    (df_2025['HomeTeam'] == 'Real Madrid') & (df_2025['pred_FTR'] == 'H')
].shape[0]

madrid_away_wins = df_2025[
    (df_2025['AwayTeam'] == 'Real Madrid') & (df_2025['pred_FTR'] == 'A')
].shape[0]

total_madrid_wins = madrid_home_wins + madrid_away_wins

print(f"El modelo predice que el Real Madrid ganará {total_madrid_wins} partidos en 2025.")

