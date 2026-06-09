import numpy as np
import pandas as pd
import xgboost as xgb
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import sklearn
import sklearn.metrics
import optuna
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb


train = pd.read_csv("/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT.csv")
test = pd.read_csv("/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv")
specimen = pd.read_csv("../input/recruitment-task-for-gdsc-ml/SPECIMEN.csv")


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.heatmap(train.corr(numeric_only=True), cmap="coolwarm", annot=False)
plt.title("correlogram)", fontsize=14)
plt.show()

train.hist(figsize=(15, 12), bins=30, edgecolor='black')
plt.suptitle("histograms", fontsize=16)
plt.show()


train.info()


column_names = train.columns
print(column_names)


train.columns = train.columns.str.strip()
train.columns = train.columns.str.replace(" ", "_")
train.columns = train.columns.str.replace("'", "")
train.columns = train.columns.str.replace("\\", "")
train.columns = train.columns.str.replace(",", "")
train.columns = train.columns.str.replace("<", "")
train.columns = train.columns.str.replace(">", "")
train.columns = train.columns.str.replace("<", "")
train.columns = train.columns.str.replace(">", "")
test.columns = test.columns.str.strip()
test.columns = test.columns.str.replace(" ", "_")
test.columns = test.columns.str.replace("'", "")
test.columns = test.columns.str.replace("\\", "")
test.columns = test.columns.str.replace(",", "")
test.columns = test.columns.str.replace("<", "")
test.columns = test.columns.str.replace(">", "")
train.columns = train.columns.str.replace("[", "")
train.columns = train.columns.str.replace("]", "")


train['maT_r'] = train['maT_r'].fillna('nan')
train["F3Ku"] = train["F3Ku"].fillna('nan')
train['MINDSPIKE_VERSION'] = train['MINDSPIKE_VERSION'].fillna('nan')


train_encoded = pd.get_dummies(train, columns=['maT_r', "F3Ku", 'MINDSPIKE_VERSION'], drop_first=True)
test_encoded = pd.get_dummies(test, columns=['maT_r', "F3Ku", 'MINDSPIKE_VERSION'], drop_first=True)

train_labels = train_encoded.columns
test_labels = test_encoded.columns

missing_in_test = set(train_labels) - set(test_labels)

for c in missing_in_test:
    test_encoded[c] = 0

missing_in_train = set(test_labels) - set(train_labels)

test_encoded = test_encoded.drop(columns=list(missing_in_train))

test_encoded = test_encoded[train_labels]


train_full = train_encoded.dropna(subset=['CORRUCYSTIC_DENSITY']).reset_index(drop=True)


train_full.info()


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

numerical_cols = train_full.select_dtypes(include=np.number).columns.tolist()

numerical_cols.remove('LOCAL_IDENTIFIER')

if 'CORRUCYSTIC_DENSITY' in numerical_cols:
    numerical_cols.remove('CORRUCYSTIC_DENSITY')

imputer = IterativeImputer(max_iter=10, random_state=42)

train_full[numerical_cols] = imputer.fit_transform(train_full[numerical_cols])

test_encoded[numerical_cols] = imputer.transform(test_encoded[numerical_cols])



y = train_full['CORRUCYSTIC_DENSITY']
X = train_full.drop(columns=['CORRUCYSTIC_DENSITY', 'LOCAL_IDENTIFIER'])

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


train_full.info()


import optuna
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
)
from sklearn.model_selection import KFold

def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "verbosity": 0,
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "early_stopping_rounds": 10
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rmse_list = []

    for train_idx, val_idx in kf.split(X):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train_cv, y_train_cv,
            eval_set=[(X_val_cv, y_val_cv)],
            
            verbose=False
        )

        y_pred = model.predict(X_val_cv)
        rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred))
        rmse_list.append(rmse)

    return np.mean(rmse_list)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

print("best trial:")
print(study.best_trial.params)


best_params = study.best_trial.params

import xgboost as xgb
model = xgb.XGBRegressor(
   **best_params
)
model.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

y_pred = model.predict(X_valid)

rmse = mean_squared_error(y_valid, y_pred, squared=False)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mean_absolute_error(y_valid, y_pred):.4f}")


X_test_aligned = test_encoded.reindex(columns=X.columns, fill_value=0)

predictions = model.predict(X_test_aligned)

submission = pd.DataFrame({
    "LOCAL_IDENTIFIER": test["LOCAL_IDENTIFIER"].astype(int),
    "CORRUCYSTIC_DENSITY": predictions.astype(float)
})

submission.to_csv("submission.csv", index=False)

print(submission.head())

