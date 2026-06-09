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


train_filepath = "/kaggle/input/playground-series-s5e12/train.csv"
test_filepath = "/kaggle/input/playground-series-s5e12/test.csv"
random_state = 42
target = "diagnosed_diabetes"
test_size = 0.2


train_df = pd.read_csv(train_filepath)
test_df = pd.read_csv(test_filepath)


import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import warnings
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


# --- 1. Prepare the data ---

df = train_df.drop('id', axis=1)

X = df.drop('diagnosed_diabetes', axis=1)
y = df['diagnosed_diabetes'].astype(int)

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# --- 2. Define the Optuna Objective Function for LightGBM ---

def objective(trial):
    """
    Objective function for Optuna to optimize LGBMClassifier hyperparameters.
    """

    param = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": 42,

        # Hyperparameters to tune
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 200),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1500),
    }

    model = LGBMClassifier(**param)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    score = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring="roc_auc",
        verbose=0
    )

    return score.mean()


# --- 3. Run the Optuna Study ---

study = optuna.create_study(direction="maximize", study_name="LightGBM_Diabetes_Prediction")

N_TRIALS = 50
print(f"Starting optimization for {N_TRIALS} trials...")

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("Optimization finished.")
print("-" * 50)


# --- 4. Print Best Hyperparameters ---

best_trial = study.best_trial
best_params = best_trial.params

print(f"✨ Best Trial AUC: {best_trial.value:.4f}")
print("Best Hyperparameters:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

print("-" * 50)


# --- 5. Train Final LightGBM Model ---

print("Training final model with best parameters...")

final_model = LGBMClassifier(
    **best_params,
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    n_jobs=-1,
    random_state=42
)

final_model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import roc_auc_score

y_pred_proba = final_model.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Final Model ROC AUC on Test Set: {test_roc_auc:.4f}")





