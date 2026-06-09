# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import label_ranking_average_precision_score
import xgboost as xgb
import optuna


# load data
train_file_path = "/kaggle/input/playground-series-s5e6/train.csv"
test_file_path = "/kaggle/input/playground-series-s5e6/test.csv"
df = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)


# label encoding
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])
df['Crop Type'] = le_crop.fit_transform(df['Crop Type'])
df['Fertilizer Name'] = le_fert.fit_transform(df['Fertilizer Name'])

test['Soil Type'] = le_soil.transform(test['Soil Type'])
test['Crop Type'] = le_crop.transform(test['Crop Type'])


# === Pipeline ===
def feature_engineering(df, scaler=None, fit_scaler=False):
    df = df.copy()
    # Humidity/Moisture Normalization
    for col in ['Humidity', 'Moisture']:
        if col in df.columns:
            df[col] = df[col] / 100

    # N/P/K log change
    for col in ['N', 'P', 'K']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])

    # Temperature normalization
    if 'Temperature' in df.columns:
        temp_col = df[['Temperature']]
        if scaler is None and fit_scaler:
            scaler = MinMaxScaler()
            df['Temperature_norm'] = scaler.fit_transform(temp_col)
        elif scaler is not None:
            df['Temperature_norm'] = scaler.transform(temp_col)
        else:
            df['Temperature_norm'] = df['Temperature'] / df['Temperature'].max()
    else:
        scaler = None

    # Category Crossover Features
    if 'Soil Type' in df.columns and 'Crop Type' in df.columns:
        df['Soil_Crop'] = df['Soil Type'].astype(str) + "_" + df['Crop Type'].astype(str)

    # Numerical cross features (products and ratios)
    pairs = [('N', 'P'), ('N', 'K'), ('P', 'K')]
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            df[f'{a}_{b}'] = df[a] * df[b]
            df[f'{a}_{b}_ratio'] = df[a] / (df[b] + 1e-6)
    return df, scaler

# Training/testing feature processing
df_fe, temp_scaler = feature_engineering(df, scaler=None, fit_scaler=True)
test_fe, _ = feature_engineering(test, scaler=temp_scaler, fit_scaler=False)

# Category Cross LabelEncoder
le_soil_crop = LabelEncoder()
df_fe['Soil_Crop'] = le_soil_crop.fit_transform(df_fe['Soil_Crop'])
test_fe['Soil_Crop'] = le_soil_crop.transform(test_fe['Soil_Crop'])

# Extracting features and labels
drop_cols = ['id', 'Fertilizer Name']
X = df_fe.drop(columns=drop_cols)
y = df_fe['Fertilizer Name']
X_test = test_fe.drop(columns=['id'])

num_classes = y.nunique()


# Optuna target function
def objective(trial):
    params = {
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'eval_metric': 'mlogloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'n_estimators': 150,   # 使用 early stopping 更科学
        'random_state': 42,
        'use_label_encoder': False,
        'verbosity': 0
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    map3_scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        y_pred_proba = model.predict_proba(X_val)
        y_val_onehot = np.zeros_like(y_pred_proba)
        y_val_onehot[np.arange(len(y_val)), y_val.values] = 1
        score = label_ranking_average_precision_score(y_val_onehot, y_pred_proba)
        map3_scores.append(score)
    return np.mean(map3_scores)


# Optuna search
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("Best MAP@3:", study.best_value)
print("Best Parameters:", study.best_params)

# Constructing the final model parameters
best_params = study.best_params
best_params.update({
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "n_estimators": 1000,
    "verbosity": 0,
    "random_state": 42,
    "use_label_encoder": False
})

# The final model is trained on all data
model = xgb.XGBClassifier(**best_params)
model.fit(X, y)


# predict
y_pred_proba = model.predict_proba(X_test)
top_3_preds = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]
top_3_labels = le_fert.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

submission = pd.DataFrame({
    "id": test["id"],
    "Fertilizer Name": [" ".join(row) for row in top_3_labels]
})
submission.to_csv("submission.csv", index=False)

