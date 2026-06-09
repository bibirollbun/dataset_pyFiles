import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Einlesen der Trainings- und Testdaten von Kaggle
train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

# Zielspalte (y) und Features (X) definieren
X = train_df.drop(['Price', 'id'], axis=1)  # Eingabedaten (ohne Zielwert und ID)
y = train_df['Price']                      # Zielvariable

# Numerische und kategoriale Spalten identifizieren
numeric_cols = ['Compartments', 'Weight Capacity (kg)']
categorical_cols = [col for col in X.columns if col not in numeric_cols]

# Fehlende Werte in numerischen Spalten durch Mittelwert ersetzen
num_imputer = SimpleImputer(strategy='mean')
X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

# Fehlende Werte in kategorialen Spalten durch häufigste Kategorie ersetzen
cat_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

# One-Hot-Encoding für kategoriale Variablen (damit sie numerisch werden)
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_cats = encoder.fit_transform(X[categorical_cols])
encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))

# Kombinieren der numerischen und kodierten Daten
X = X.drop(columns=categorical_cols)
X = pd.concat([X.reset_index(drop=True), encoded_cat_df.reset_index(drop=True)], axis=1)

# Aufteilen der Daten in Training und Test (z.B. 80% / 20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Einfaches Regressionsmodell trainieren
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Vorhersage auf Validierungsdaten
y_pred = lr_model.predict(X_val)

# Bewertung des Modells mit Standard-Metriken
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R²-Score: {r2:.4f}")

# Scatterplot: Tatsächliche vs. vorhergesagte Werte
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, alpha=0.6, color='teal')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.title('Tatsächliche vs. vorhergesagte Preise')
plt.xlabel('Tatsächlicher Preis')
plt.ylabel('Vorhergesagter Preis')
plt.grid(True)
plt.show()

# Vorhersagen für den echten Testdatensatz erzeugen
X_test = test_df.drop('id', axis=1)
X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])
encoded_test = encoder.transform(X_test[categorical_cols])
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols))
X_test = X_test.drop(columns=categorical_cols)
X_test = pd.concat([X_test.reset_index(drop=True), encoded_test_df.reset_index(drop=True)], axis=1)

# Prediction für Testdaten und Submission vorbereiten
test_pred = lr_model.predict(X_test)
submission = test_df[['id']].copy()
submission['Price'] = test_pred
submission.to_csv('/kaggle/working/submission.csv', index=False)
submission.head()


