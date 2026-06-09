""" 
Goal: Perform HP Tuning on RF, XGB, CATB, and LGB separately.
 
Author: Rudra Prasad Bhuyan
"""
print("")


!nvidia-smi


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, 
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc
)
from sklearn.utils import class_weight

from sklearn import set_config
set_config(display="diagram")

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier


import warnings
warnings.simplefilter('ignore')


SUB_PATH    = r"/kaggle/input/playground-series-s5e11/sample_submission.csv" 
TRAIN_PATH  = r"/kaggle/input/playground-series-s5e11/train.csv"
TEST_PATH   = r"/kaggle/input/playground-series-s5e11/test.csv"

RANDOM_SEED = 42
BATCH_SIZE  = 4096
EPOCHS      = 100
VALID_SIZE  = 0.2
MODEL_OUT   = "best_ann.h5"
SUB_OUT     = "submission.csv"

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
sub_df = pd.read_csv(SUB_PATH)


TARGET = "loan_paid_back"
ID_COL = "id"

NUMERIC_COLS = [
    "annual_income", "debt_to_income_ratio", "credit_score",
    "loan_amount", "interest_rate"
]
CAT_COLS = [
    "gender", "marital_status", "education_level",
    "employment_status", "loan_purpose", "grade_subgrade"
]


numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore",))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CAT_COLS)
    ],
    remainder="drop"
)

preprocessor


X_all = train_df[NUMERIC_COLS + CAT_COLS]
y_all = train_df[TARGET].astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=VALID_SIZE, 
    random_state=RANDOM_SEED, stratify=y_all
)


preprocessor.fit(X_train)

X_train_p = preprocessor.transform(X_train)
X_val_p   = preprocessor.transform(X_val)
X_test_p  = preprocessor.transform(test_df[NUMERIC_COLS + CAT_COLS])

print("Preprocessed shapes:", X_train_p.shape, X_val_p.shape, X_test_p.shape)


def objective(trial):
    # Choose model
    model_name = trial.suggest_categorical("model", ["xgb", "lgbm", "catb", "rf"])
    
    if model_name == "xgb":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1
        }
        # GPU explicitly enabled
        model = XGBClassifier(**params, tree_method="gpu_hist", predictor="gpu_predictor")

    elif model_name == "lgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", -1, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "n_jobs": -1
        }
        # GPU explicitly enabled
        model = LGBMClassifier(**params, device="gpu")

    elif model_name == "catb":
        params = {
            "iterations": trial.suggest_int("iterations", 200, 800),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
            "random_state": 42
        }
        # GPU explicitly enabled
        model = CatBoostClassifier(**params, task_type="GPU", verbose=0)

    elif model_name == "rf":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": 42,
            "n_jobs": -1
        }
        # RandomForest is CPU-only
        model = RandomForestClassifier(**params)
    
    # --- Train ---
    model.fit(X_train_p, y_train)

    # --- Predict probabilities ---
    preds_val_proba = model.predict_proba(X_val_p)[:, 1]

    # --- Compute ROC-AUC ---
    auc_score = roc_auc_score(y_val, preds_val_proba)
    
    return auc_score


# Run optimization for each model separately

models = ["xgb", "lgbm", "catb", "rf"]
best_models = {}

for m in models:
    print(f"\nðŸ”¹ Running Optuna for {m.upper()}...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial), n_trials=30, show_progress_bar=True)
    
    best_params = study.best_params
    print(f"\n  Best {m.upper()} AUC:", study.best_value)
    print("Best Parameters:", best_params)

    # Rebuild best model (with fixed model type)
    if m == "xgb":
        best_model = XGBClassifier(**{k:v for k,v in best_params.items() if not k=="model"},
                                   use_label_encoder=False, eval_metric="logloss", random_state=42)
    elif m == "lgbm":
        best_model = LGBMClassifier(**{k:v for k,v in best_params.items() if not k=="model"}, random_state=42)
    elif m == "catb":
        best_model = CatBoostClassifier(**{k:v for k,v in best_params.items() if not k=="model"}, verbose=0, random_state=42)
    elif m == "rf":
        best_model = RandomForestClassifier(**{k:v for k,v in best_params.items() if not k=="model"}, random_state=42)

    # Retrain on full train + val
    best_model.fit(np.vstack((X_train_p, X_val_p)), np.concatenate((y_train, y_val)))
    best_models[m] = (best_model, study.best_value, best_params)


for m, (model, auc_val, params) in best_models.items():
    print(f"\nðŸ”¸ Finalizing {m.upper()} model | Best AUC: {auc_val:.4f}")
    
    # Validation ROC-AUC and ROC curve
    val_pred_proba = model.predict_proba(X_val_p)[:, 1]
    auc_score = roc_auc_score(y_val, val_pred_proba)
    print(f"{m.upper()} Validation AUC: {auc_score:.4f}")

    # Predict on test data
    test_pred_proba = model.predict_proba(X_test_p)[:, 1]

    # Create submission CSV
    submission = pd.DataFrame({
        "id": test_df["id"],  
        "target": test_pred_proba
    })
    
    file_name = f"submission_{m}.csv"
    submission.to_csv(file_name, index=False)
    print(f"Saved {file_name}")

