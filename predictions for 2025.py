import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
#from funtions import prepare_data
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
def prepare_data(data, numeric_cols, bets_cols):
    htr = data['HTR'].astype('category').cat.codes
    home_teams = data['HomeTeam'].astype('category').cat.codes
    away_teams = data['AwayTeam'].astype('category').cat.codes
    
    X_with_bets = pd.concat([
        pd.DataFrame(home_teams, columns=['HomeTeam']),
        pd.DataFrame(away_teams, columns=['AwayTeam']),
        pd.DataFrame(htr, columns=['HTR']),
        data[numeric_cols].reset_index(drop=True),
        data[bets_cols].reset_index(drop=True)
    ], axis=1)
    
    X_without_bets = pd.concat([
        pd.DataFrame(home_teams, columns=['HomeTeam']),
        pd.DataFrame(away_teams, columns=['AwayTeam']),
        pd.DataFrame(htr, columns=['HTR']),
        data[numeric_cols].reset_index(drop=True)
    ], axis=1)
    
    return X_with_bets, X_without_bets


data_until_2024_path = r"C:\Users\Usuario\Desktop\cositas en python\football\data-in-football\datasets\Dataset from 2019 to 2024.csv"
data_until_2024 = pd.read_csv(data_until_2024_path,sep=';')


data_2025_path = r"C:\Users\Usuario\Desktop\cositas en python\football\data-in-football\datasets\LaLiga_24_25_transform.csv"
data_2025 = pd.read_csv(data_2025_path, sep=';')

for df in [data_until_2024, data_2025]:
    df['HomeTeam'] = df['HomeTeam'].astype(str).str.strip().str.upper()
    df['AwayTeam'] = df['AwayTeam'].astype(str).str.strip().str.upper()


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

## Initialize the model OvO:

model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)

y_pred_ovo = model_ovo.predict(X_test)


## Evaluation of the model
print("One-vs-One (OvO) Strategy:")

# Calculate the metrics
precision = precision_score(y_test, y_pred_ovo, average=None)
recall = recall_score(y_test, y_pred_ovo, average=None)
f1 = f1_score(y_test, y_pred_ovo, average=None)

# Creation of a evaluation table
labels = ['Home wins', 'Draw', 'Away Wins']
width = 15  # ancho de cada columna para alineación
header = "".join(f"{label:<{width}}" for label in labels)
precision_row = "".join(f"{val:<{width}.4f}" for val in precision)
recall_row = "".join(f"{val:<{width}.4f}" for val in recall)
f1_row = "".join(f"{val:<{width}.4f}" for val in f1)

print("Model Evaluation Metrics (H/D/A):\n")
print(f"{'Results':<10}{header}")
print(f"{'Precision':<10}{precision_row}")
print(f"{'Recall':<10}{recall_row}")
print(f"{'F1-score':<10}{f1_row}")

print(f'\nAccuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%')
print(confusion_matrix(y_test, y_pred_ovo))



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

# Create the classification table that our prediction predict:

teams = pd.Index(data_2025['HomeTeam']).union(data_2025['AwayTeam'])
teams = teams.dropna()
teams = teams.unique()

predict_table = pd.DataFrame(
    0, index=pd.Index(teams, name="Team"), 
    columns=['Points', 'Played', 'W', 'D', 'L']
)

# Convert 'FTR' to letters again (H, D, A)
result_mapping = dict(enumerate(data_until_2024['FTR'].astype('category').cat.categories))
pred_results = pd.Series(y_pred_ovo).map(result_mapping)   


# for making the table, we will add the points and stats using the match results
for index, row in data_2025.iterrows():
    home = row['HomeTeam']
    away = row['AwayTeam']
    pred = pred_results.iloc[index]  # Predicción para ese partido


    predict_table.loc[home, 'Played'] += 1
    predict_table.loc[away, 'Played'] += 1

    # If home team wins
    if pred == 'H':
        predict_table.loc[home, 'Points'] += 3
        predict_table.loc[home, 'W'] += 1
        predict_table.loc[away, 'L'] += 1

    # If there is a Draw
    elif pred == 'D':
        predict_table.loc[home, 'Points'] += 1
        predict_table.loc[away, 'Points'] += 1
        predict_table.loc[home, 'D'] += 1
        predict_table.loc[away, 'D'] += 1
    
    # If away team wins
    else:  # 'A'
        predict_table.loc[away, 'Points'] += 3
        predict_table.loc[away, 'W'] += 1
        predict_table.loc[home, 'L'] += 1

predict_table = predict_table.reset_index().rename(columns={'index': 'Team'})
predict_table = predict_table.rename(columns={'HomeTeam': 'Team'}) if 'HomeTeam' in predict_table.columns else predict_table


predict_table = predict_table.sort_values(by=['Points', 'W'], ascending=False)

print('\n--- PREDICTION WITH THE MODEL ---')
print(predict_table)


# Create the real classification table of the 2024/2025 season, with the data:

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

classification_table = classification_table.reset_index().rename(columns={'index': 'Team'})
classification_table = classification_table.rename(columns={'HomeTeam': 'Team'}) if 'HomeTeam' in classification_table.columns else classification_table


classification_table = classification_table.sort_values(by=['Points', 'W'], ascending=False)

print('\n--- CLASSIFICATION SEASON 2024/2025 ---')
print(classification_table)


## Error table:
print('\nError table:')

# Set the teams as index
predict_fixed = predict_table.set_index("Team")
classif_fixed = classification_table.set_index("Team")
classif_fixed = classif_fixed.reindex(index=predict_fixed.index)


common_cols = ['Points', 'W', 'D', 'L']
differences_table = predict_fixed[common_cols] - classif_fixed[common_cols]
differences_table = differences_table.reset_index()

print(differences_table)
