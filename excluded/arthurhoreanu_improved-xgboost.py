import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Daten einlesen
train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
extra_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')

# Trainings- und Zusatzdaten kombinieren
train_df = pd.concat([train_df, extra_df], ignore_index=True)

# Zielvariable und Merkmale definieren
X = train_df.drop(['Price', 'id'], axis=1)
y = train_df['Price']

# Numerische und kategoriale Spalten identifizieren
numeric_cols = ['Compartments', 'Weight Capacity (kg)']
categorical_cols = [col for col in X.columns if col not in numeric_cols]

# Fehlende Werte imputieren
num_imputer = SimpleImputer(strategy='mean')
X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

# One-Hot-Encoding für kategoriale Variablen
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_cats = encoder.fit_transform(X[categorical_cols])
encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))

# Kombinieren der numerischen und kodierten Daten
X = X.drop(columns=categorical_cols)
X = pd.concat([X.reset_index(drop=True), encoded_cat_df.reset_index(drop=True)], axis=1)

# BASELINE-MODELL (für Submission)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_val)

# Evaluation
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("Baseline XGBoost:")
print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

# Vorbereitung der Testdaten für die SUBMISSION (keine neuen Features!)
X_test = test_df.drop(columns=['id'])
X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])
encoded_test = encoder.transform(X_test[categorical_cols])
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols))
X_test = X_test.drop(columns=categorical_cols)
X_test = pd.concat([X_test.reset_index(drop=True), encoded_test_df.reset_index(drop=True)], axis=1)

# Submission speichern
test_pred = xgb_model.predict(X_test)
submission = test_df[['id']].copy()
submission['Price'] = test_pred
submission.to_csv('/kaggle/working/submission.csv', index=False)

# Feature Engineering separat evaluieren (aber nicht für Submission)
X_fe = X.copy()
X_fe["Weight_per_Compartment"] = X_fe["Weight Capacity (kg)"] / (X_fe["Compartments"] + 1)
X_train_fe, X_val_fe, y_train_fe, y_val_fe = train_test_split(X_fe, y, test_size=0.2, random_state=42)
model_fe = XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
model_fe.fit(X_train_fe, y_train_fe)
y_pred_fe = model_fe.predict(X_val_fe)

mse_fe = mean_squared_error(y_val_fe, y_pred_fe)
rmse_fe = np.sqrt(mse_fe)
mae_fe = mean_absolute_error(y_val_fe, y_pred_fe)
r2_fe = r2_score(y_val_fe, y_pred_fe)

print("\nMit neuem Feature (Weight_per_Compartment):")
print(f"MSE: {mse_fe:.2f}, RMSE: {rmse_fe:.2f}, MAE: {mae_fe:.2f}, R²: {r2_fe:.4f}")

# Hyperparameter-Tuning separat evaluieren (nicht für Submission)
model_tuned = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
model_tuned.fit(X_train_fe, y_train_fe)
y_pred_tuned = model_tuned.predict(X_val_fe)

mse_tuned = mean_squared_error(y_val_fe, y_pred_tuned)
rmse_tuned = np.sqrt(mse_tuned)
mae_tuned = mean_absolute_error(y_val_fe, y_pred_tuned)
r2_tuned = r2_score(y_val_fe, y_pred_tuned)

print("\nMit Hyperparameter-Tuning (max_depth=6):")
print(f"MSE: {mse_tuned:.2f}, RMSE: {rmse_tuned:.2f}, MAE: {mae_tuned:.2f}, R²: {r2_tuned:.4f}")


