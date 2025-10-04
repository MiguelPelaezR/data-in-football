import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score




df = pd.read_csv('datasets/LaLiga_24_25.csv')

#Upon review, it was observed that all columns containing missing values are related to betting data; therefore, these variables will be excluded from the analysis.
df = df.dropna(axis=1)

'''titles = list(df.columns)
print(titles)'''

# Lets divide the data frame in to some features:
match_data = df[['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]
FTR_data = df['FTR'].astype('category').cat.codes

#df_total = pd.concat([match_data, FTR_data], axis=1)
X = match_data.select_dtypes(include=[np.number])  # Exclude the team names 
y = FTR_data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

### Training logistic regression model using One-vs-All


model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)

# Make predictions on the test set
y_pred_ova = model_ova.predict(X_test)

# Calculate accuracy
print("One-vs-All (OvA) Strategy:")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")


### Training logistic regression model using One-vs-One

model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)

# Make predictions on the test set
y_pred_ovo = model_ovo.predict(X_test)
# Calculate accuracy
print("One-vs-One (OvO) Strategy:")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")


plt.figure(figsize=(10, 5))
sns.countplot(x=y_test, hue=y_pred_ovo, palette='Set1')
plt.title('Predicted vs Actual Obesity Levels')
plt.xlabel('Actual Obesity Level')
plt.ylabel('Count')
plt.legend(title='Predicted Obesity Level', loc='upper right')
plt.show()

feature_importance = np.mean(np.abs(model_ova.coef_), axis=0)
plt.barh(X.columns, feature_importance)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()














