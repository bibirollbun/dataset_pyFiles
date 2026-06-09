import numpy as np
import pandas as pd
import os

import pylab
import scipy.stats as stats

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns


# Die Daten werden hochgeladet
train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


# Man wird uberpruft, falls NA Daten vorhanden sind
train_data.isna().sum()

train_data.dropna(inplace=True)


# id Spalte wird entworfen
train_data.drop(columns=['id'], inplace=True)
test_data.drop(columns=['id'], inplace=True)


# NA Daten werden ersetzt
for col in test_data:
    if test_data[col].dtype == 'object':
        test_data[col].fillna('not listed', inplace=True)  # fur categorical Daten
    else:
        test_data[col].fillna(-1, inplace=True)  # fur numerische Daten


# Encoding
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

for col in train_data.select_dtypes(include=['object']).columns:
    train_data[col] = encoder.fit_transform(train_data[col].values.reshape(-1, 1))
    test_data[col] = encoder.transform(test_data[col].values.reshape(-1, 1))


# Wir trennen die Merkmale (X) und das Ziel (y)
target = train_data.pop('Price').copy() 
features = train_data
test_features = test_data


# Train und Test split
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.1, shuffle=True, random_state=42)


# ElasticNet Model trainieren
model = ElasticNet(random_state=42).fit(X_train, y_train)
model.score(X_train, y_train)


y_pred_val = model.predict(X_val)


# # Berechnet den Root Mean Squared Error (RMSE), der die durchschnittliche Abweichung zwischen den tatsächlichen 
# und den vorhergesagten Werten darstellt und eine intuitivere Maßzahl für die Modellgenauigkeit liefert.

mse = mean_squared_error(y_val, y_pred_val)
rmse = np.sqrt(mse)
rmse


# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred_val, edgecolors="k", alpha=0.6)
plt.plot([target.min(), target.max()], [target.min(), target.max()], "r--", lw=2)
plt.xlabel("Eigentliche Werte")
plt.ylabel("Vorhersage")
plt.title("Eigentliche Werte vs. Vorhersage")
plt.show()


# Vorhersage fur Testdaten
pred_test = model.predict(test_features)


# Submission File
submission['Price'] = pred_test
submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
submission

