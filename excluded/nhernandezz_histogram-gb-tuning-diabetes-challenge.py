import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import optuna


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
numeric_df = df.select_dtypes(include=["number"])
numeric_df["ldl_hdl_ratio"] = numeric_df["ldl_cholesterol"] / (numeric_df["hdl_cholesterol"] + 1e-6)
numeric_df["cholesterol_ratio"] = numeric_df["cholesterol_total"] / (numeric_df["hdl_cholesterol"] + 1e-6)
numeric_df["activity_bmi"] = numeric_df["physical_activity_minutes_per_week"] / (numeric_df["bmi"] + 1e-6)
numeric_df["age_bmi"] = numeric_df["age"] * numeric_df["bmi"]
numeric_df["age_activity"] = numeric_df["age"] * numeric_df["physical_activity_minutes_per_week"]
numeric_df["age_triglycerides"] = numeric_df["age"] * numeric_df["triglycerides"]
numeric_df["high_bmi"] = (numeric_df["bmi"] > 30).astype(int)
numeric_df["high_triglycerides"] = (numeric_df["triglycerides"] > 150).astype(int)
numeric_df["insulin_resistance_proxy"] = (numeric_df["bmi"] * numeric_df["triglycerides"] / (numeric_df["physical_activity_minutes_per_week"] + 1))
numeric_df["metabolic_risk_score"] = (
    0.4 * numeric_df["bmi"] +
    0.3 * numeric_df["triglycerides"] +
    0.2 * numeric_df["age"] -
    0.3 * numeric_df["physical_activity_minutes_per_week"]
)
numeric_df.head()


features = [
    "family_history_diabetes",
    "physical_activity_minutes_per_week",
    "activity_bmi",
    "age_bmi",
    "age_triglycerides",
    "age_activity",
    "age",
    "ldl_hdl_ratio",
    "triglycerides",
    "cholesterol_ratio",
    "bmi",
    "insulin_resistance_proxy",
    "metabolic_risk_score"
]

X = numeric_df[features]
y = numeric_df["diagnosed_diabetes"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, 
                                                random_state=42,stratify=y_test)


def objective(trial):
    
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "max_iter": trial.suggest_int("max_iter", 200, 1000),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        "max_bins": trial.suggest_int("max_bins", 128, 255),
        "random_state": 42
    }
    
    model = HistGradientBoostingClassifier(**params)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(X_val)[:, 1]
        
        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)
    
    return np.mean(aucs)


#study = optuna.create_study(direction="maximize")
#study.optimize(objective, n_trials=20, show_progress_bar=True)


'''
paramsLR = study.best_params
hbc = HistGradientBoostingClassifier(**paramsLR)
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])
hbc.fit(X_train_full, y_train_full)     
y_score = hbc.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_score)
auc
'''


X_train_full = pd.concat([X_train, X_val, X_test])
y_train_full = pd.concat([y_train, y_val, y_test])
hbc = HistGradientBoostingClassifier(**{'learning_rate': 0.04129003713472085,
 'max_depth': 7,
 'max_iter': 863,
 'min_samples_leaf': 49,
 'l2_regularization': 0.159201740522381,
 'max_bins': 237,
 'random_state':777})
hbc.fit(X_train_full, y_train_full) 


test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
numeric_df = test.select_dtypes(include=["number"])
numeric_df["ldl_hdl_ratio"] = numeric_df["ldl_cholesterol"] / (numeric_df["hdl_cholesterol"] + 1e-6)
numeric_df["cholesterol_ratio"] = numeric_df["cholesterol_total"] / (numeric_df["hdl_cholesterol"] + 1e-6)
numeric_df["activity_bmi"] = numeric_df["physical_activity_minutes_per_week"] / (numeric_df["bmi"] + 1e-6)
numeric_df["age_bmi"] = numeric_df["age"] * numeric_df["bmi"]
numeric_df["age_activity"] = numeric_df["age"] * numeric_df["physical_activity_minutes_per_week"]
numeric_df["age_triglycerides"] = numeric_df["age"] * numeric_df["triglycerides"]
numeric_df["high_bmi"] = (numeric_df["bmi"] > 30).astype(int)
numeric_df["high_triglycerides"] = (numeric_df["triglycerides"] > 150).astype(int)
numeric_df["insulin_resistance_proxy"] = (numeric_df["bmi"] * numeric_df["triglycerides"] / (numeric_df["physical_activity_minutes_per_week"] + 1))
numeric_df["metabolic_risk_score"] = (
    0.4 * numeric_df["bmi"] +
    0.3 * numeric_df["triglycerides"] +
    0.2 * numeric_df["age"] -
    0.3 * numeric_df["physical_activity_minutes_per_week"]
)
full = numeric_df


X = full[features]
y_score = hbc.predict_proba(X)[:,1]
test['diagnosed_diabetes'] = y_score
submit = test[['id','diagnosed_diabetes']]
submit = submit.set_index('id')
submit.to_csv('submission.csv')
submit

