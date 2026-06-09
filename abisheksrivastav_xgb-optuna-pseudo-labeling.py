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


# Personality Classification - XGB + Optuna + Pseudo-Labeling

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

# Label encode target
y = train["Personality"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)
train["Personality_encoded"] = y_encoded

# Prepare features
X = train.drop(columns=["id", "Personality", "Personality_encoded"])
y = train["Personality_encoded"]
X_test = test.drop(columns=["id"])

# Ordinal encode categorical columns
combined = pd.concat([X, X_test], axis=0)
cat_cols = combined.select_dtypes(include="object").columns.tolist()
encoder = OrdinalEncoder()
combined[cat_cols] = encoder.fit_transform(combined[cat_cols])

X = combined.iloc[:len(X)].reset_index(drop=True)
X_test = combined.iloc[len(X):].reset_index(drop=True)

# Pseudo-labeling setup
pseudo_threshold = 0.99  # use only very confident predictions
pseudo_preds = np.zeros(len(X_test))
pseudo_confidences = np.zeros(len(X_test))

# Optuna objective function
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "eta": trial.suggest_float("eta", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.8, 1.2),
        "random_state": 42,
        "verbosity": 0
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(params, dtrain, num_boost_round=1000,
                          evals=[(dval, "valid")],
                          early_stopping_rounds=20,
                          verbose_eval=False)

        oof_preds[val_idx] = model.predict(dval) > 0.5

    acc = accuracy_score(y, oof_preds)
    return acc

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print("Best trial:", study.best_trial.params)

# Train with best params
params = study.best_trial.params
params.update({
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "verbosity": 0
})

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

pseudo_X = []
pseudo_y = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(params, dtrain, num_boost_round=1000,
                      evals=[(dval, "valid")],
                      early_stopping_rounds=20, verbose_eval=False)

    val_preds = model.predict(dval) > 0.5
    oof_preds[val_idx] = val_preds

    fold_test_preds = model.predict(dtest)
    test_preds += fold_test_preds / skf.n_splits

    # Pseudo-label high confidence predictions
    for i, prob in enumerate(fold_test_preds):
        if prob > pseudo_threshold:
            pseudo_X.append(X_test.iloc[i])
            pseudo_y.append(1)
        elif prob < 1 - pseudo_threshold:
            pseudo_X.append(X_test.iloc[i])
            pseudo_y.append(0)

# Evaluate
cv_acc = accuracy_score(y, oof_preds)
print(f"Cross-Validation Accuracy: {cv_acc:.4f}")

# Retrain on combined train + pseudo data
if pseudo_X:
    pseudo_X_df = pd.DataFrame(pseudo_X).reset_index(drop=True)
    pseudo_y_df = pd.Series(pseudo_y)
    X_combined = pd.concat([X, pseudo_X_df], axis=0).reset_index(drop=True)
    y_combined = pd.concat([pd.Series(y), pseudo_y_df], axis=0).reset_index(drop=True)

    dtrain_final = xgb.DMatrix(X_combined, label=y_combined)
    dtest_final = xgb.DMatrix(X_test)

    final_model = xgb.train(params, dtrain_final, num_boost_round=study.best_trial.number)
    final_preds = (final_model.predict(dtest_final) > 0.5).astype(int)
else:
    final_preds = (test_preds > 0.5).astype(int)

# Submission
submission["Personality"] = le.inverse_transform(final_preds)
submission.to_csv("submission.csv", index=False)
print("submission.csv saved")


