import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train.head()


test.head()


train.describe()


train.isnull().sum()


test.isnull().sum()


sns.countplot(x=train['rainfall'], palette="viridis")
plt.title("Rainfall Distribution (0: No, 1: Yes)")
plt.show()



# Correlation Matrix
plt.figure(figsize=(12,8))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.show()


train.hist(figsize=(12, 9), bins=15)
plt.suptitle("Feature Distributions")
plt.show()



plt.figure(figsize=(10, 5))
sns.boxplot(x=train['rainfall'], y=train['humidity'], palette="coolwarm")
plt.title("Rainfall and Humidity Relationship")
plt.show()


plt.figure(figsize=(10, 5))
sns.boxplot(x=train['rainfall'], y=train['temparature'], palette="coolwarm")
plt.title("Rainfall and Temperature Relationship")
plt.show()


# Rainfall Distribution by Days
plt.figure(figsize=(12, 5))
sns.lineplot(x=train['day'], y=train['rainfall'], marker="o")
plt.title("Rainfall by Days")
plt.xlabel("Day")
plt.ylabel("Rainfall (0 = No, 1 = Yes)")
plt.show()



y = train['rainfall']  
X = train.drop(columns=['id', 'day', 'rainfall'])  
test_data = test.drop(columns=['id', 'day'])



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_data = scaler.transform(test_data)



rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_val)



xgb_model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_val)




print("Random Forest Sonuçları:")
print(classification_report(y_val, rf_preds))
print("Accuracy:", accuracy_score(y_val, rf_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_val, rf_preds))



print("XGBoost Sonuçları:")
print(classification_report(y_val, xgb_preds))
print("Accuracy:", accuracy_score(y_val, xgb_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_val, xgb_preds))



final_preds = xgb_model.predict(test_data)
test['rainfall'] = final_preds
print("Predictions:")
print(test[['id', 'rainfall']].head(10))  
test[['id', 'rainfall']].to_csv("submission.csv", index=False)





