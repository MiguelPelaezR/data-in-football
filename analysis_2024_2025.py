import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('datasets/LaLiga_24_25.csv')

#Upon review, it was observed that all columns containing missing values are related to betting data; therefore, these variables will be excluded from the analysis.
df = df.dropna(axis=1)

titles = list(df.columns)
print(titles)

# Lets divide the data frame in to some features:
match_data = df[['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]
FTR_data = pd.get_dummies(df[['FTR']])

df_total = pd.concat([match_data, FTR_data], axis=1)

# Lets see some characteristics:
print(match_data.describe())
print(df_total.corr(numeric_only=True).T)

### LOGISTIC REGRESSION
X = match_data.select_dtypes(include=[np.number])  # Exclude the team names 
y = df['FTR'].astype('category').cat.codes

# Estandarizar
std_scaler = preprocessing.StandardScaler()
X_std = std_scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

# Modelo de regresión logística multinomial
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)

# Evaluación
print("Train accuracy:", log_reg.score(X_train, y_train))
print("Test accuracy:", log_reg.score(X_test, y_test))
print("Coeficientes por clase:", log_reg.coef_)

feature_importance = np.mean(np.abs(log_reg.coef_), axis=0)
plt.barh(X.columns, feature_importance)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()


### RAMDOM FOREST
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

print("Train accuracy:", rf.score(X_train, y_train))
print("Test accuracy:", rf.score(X_test, y_test))

y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))









