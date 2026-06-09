import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd

train = pd.read_csv("/kaggle/input/sparta-2024-data-science-competition/train.csv", nrows=80000)
test = pd.read_csv("/kaggle/input/sparta-2024-data-science-competition/test.csv")
submission = pd.read_csv("/kaggle/input/sparta-2024-data-science-competition/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


from sklearn.preprocessing import LabelEncoder

train.fillna(-999, inplace=True)
test.fillna(-999, inplace=True)

for col in train.columns:
    if train[col].dtype == "object":
        le = LabelEncoder()
        le.fit(list(train[col].astype(str)) + list(test[col].astype(str)))
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))


from sklearn.model_selection import train_test_split

X = train.drop("price", axis=1)
y = train["price"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np

def objective_xgb(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'tree_method': 'hist',   # lebih hemat RAM
    }

    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=50,
              verbose=False)

    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective_xgb, n_trials=20)

print("Best params (XGBoost):", study_xgb.best_params)
print("Best RMSE (XGBoost):", study_xgb.best_value)


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

rows_to_read = 80000
skip = sorted(np.random.choice(range(1, 300001), 300000-rows_to_read, replace=False))
train_sample = pd.read_csv(
    "/kaggle/input/sparta-2024-data-science-competition/train.csv",
    skiprows=skip
)

test_full = pd.read_csv("/kaggle/input/sparta-2024-data-science-competition/test.csv")
submission = pd.read_csv("/kaggle/input/sparta-2024-data-science-competition/sample_submission.csv")

train_sample.fillna(-999, inplace=True)
test_full.fillna(-999, inplace=True)

for col in train_sample.columns:
    if train_sample[col].dtype == "object":
        le = LabelEncoder()
        le.fit(list(train_sample[col].astype(str)) + list(test_full[col].astype(str)))
        train_sample[col] = le.transform(train_sample[col].astype(str))
        test_full[col] = le.transform(test_full[col].astype(str))

X_full = train_sample.drop("price", axis=1)
y_full = train_sample["price"]

for df in [X_full, test_full]:
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')

dtrain = xgb.DMatrix(X_full, label=y_full)
dtest = xgb.DMatrix(test_full)

best_params = study_xgb.best_params
best_params['n_estimators'] = min(best_params['n_estimators'], 400)  # pohon <= 400
best_params['max_depth'] = min(best_params['max_depth'], 6)          # kedalaman <= 6

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'learning_rate': best_params['learning_rate'],
    'max_depth': best_params['max_depth'],
    'subsample': best_params['subsample'],
    'colsample_bytree': best_params['colsample_bytree'],
    'random_state': 42
}

final_model = xgb.train(params, dtrain, num_boost_round=best_params['n_estimators'])

test_preds = final_model.predict(dtest)

submission["price"] = test_preds
submission.to_csv("/kaggle/working/submission_xgb_ultrasafe_small.csv", index=False)
print("✅ submission_xgb_ultrasafe_small.csv berhasil dibuat")

