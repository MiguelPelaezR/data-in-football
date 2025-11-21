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


# Lets divide the data frame in to some features:
variables = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A', 'PSH', 'PSD', 'PSA', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA']
X_train = df_2024[variables]
y_train = df_2024['FTR'].astype('category').cat.codes

std_scaler = preprocessing.StandardScaler()
X_train = std_scaler.fit_transform(X_train)

X_test = df_2025[['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A', 'PSH', 'PSD', 'PSA', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA']]
y_test = df_2025['FTR'].astype('category').cat.codes

X_test = std_scaler.fit_transform(X_test)


### With One vs All method:


model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)

# Make predictions on the test set
y_pred_ova = model_ova.predict(X_test)

# Calculate accuracy
print("One-vs-All (OvA) Strategy:")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")


feature_importance = np.mean(np.abs(model_ova.coef_), axis=0)
plt.barh(variables, feature_importance)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()



### With Logistic regresion:

# Modelo de regresión logística multinomial
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)

# Evaluación
print("Train accuracy:", log_reg.score(X_train, y_train))
print("Test accuracy:", log_reg.score(X_test, y_test))
print("Coeficientes por clase:", log_reg.coef_)

feature_importance = np.mean(np.abs(log_reg.coef_), axis=0)
plt.barh(variables, feature_importance)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()



