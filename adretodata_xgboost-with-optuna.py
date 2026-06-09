import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error

# === Load Data ===
train_df = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv")
test_df = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv")


# === Preprocessing ===
train_df['date'] = pd.to_datetime(train_df['date'])
train_df['letters'] = train_df['plate'].str[:1] + train_df['plate'].str[4:5] + train_df['plate'].str[5:6]
train_df['numbers'] = train_df['plate'].str[1:4].astype(int)
train_df['region'] = train_df['plate'].str[6:]



# Fungsi cek instansi pemerintah
from supplemental_english import REGION_CODES, GOVERNMENT_CODES
def check_gov_related(row):
    key = (row['letters'], (row['numbers'], row['numbers']), row['region'])
    return key in GOVERNMENT_CODES

train_df['gov_related'] = train_df.apply(check_gov_related, axis=1)

# === One-Hot Encoding Region ===
encoder = OneHotEncoder(handle_unknown='ignore')
region_encoded = encoder.fit_transform(train_df[['region']]).toarray()
region_cols = encoder.get_feature_names_out(['region'])
region_df = pd.DataFrame(region_encoded, columns=region_cols)




# Gabungkan data
train_df = pd.concat([train_df, region_df], axis=1)

# === Pilih Fitur ===
features = ['numbers', 'gov_related'] + list(region_cols)
X = train_df[features]
y = train_df['price']

# === Split Train & Validation ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Fungsi SMAPE ===
def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100




# === Hyperparameter Tuning dengan Optuna ===
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    model = xgb.XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

    y_pred = model.predict(X_val)
    return smape(y_val, y_pred)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)




# === Train Model Final dengan Hyperparameter Optimal ===
best_params = study.best_params
model = xgb.XGBRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

# === Evaluasi SMAPE ===
y_pred = model.predict(X_val)
smape_score = smape(y_val, y_pred)
print(f"Best SMAPE Score: {smape_score:.2f}%")



# ==== Prediksi pada Test Set ====
test_df['letters'] = test_df['plate'].str[:1] + test_df['plate'].str[4:5] + test_df['plate'].str[5:6]
test_df['numbers'] = test_df['plate'].str[1:4].astype(int)
test_df['region'] = test_df['plate'].str[6:]
test_df['gov_related'] = test_df.apply(check_gov_related, axis=1)

test_region_encoded = encoder.transform(test_df[['region']]).toarray()
test_region_df = pd.DataFrame(test_region_encoded, columns=region_cols)
test_df = pd.concat([test_df, test_region_df], axis=1)

test_df['price'] = model.predict(test_df[features])


test_df[['id', 'price']].to_csv("./best_smape.csv", index=False)

