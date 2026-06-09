import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier

import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ASK - Define the problem and goal
# Goal: Predict rainfall occurrence using various weather features.


# PREPARE - Load and clean data
train_path = "/kaggle/input/playground-series-s5e3/train.csv"
test_path  = "/kaggle/input/playground-series-s5e3/test.csv"
original_path = "/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv"

df_train = pd.read_csv(train_path, index_col="id")
df_test  = pd.read_csv(test_path,  index_col="id")
df_original = pd.read_csv(original_path)

df_original.columns = [col.strip() for col in df_original.columns]
df_original['rainfall'] = df_original['rainfall'].map({'yes': 1, 'no': 0})
df_original.dropna(inplace=True)

df_train = pd.concat([df_train, df_original], axis=0, ignore_index=True)
df_train.fillna(0, inplace=True)

df_test.fillna(df_test.median(), inplace=True)


# PROCESS - Feature Engineering & Preprocessing
class OutlierCapTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_q=0.01, upper_q=0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q
    
    def fit(self, X, y=None):
        self.lower_bounds_ = X.quantile(self.lower_q)
        self.upper_bounds_ = X.quantile(self.upper_q)
        return self
    
    def transform(self, X):
        return X.clip(self.lower_bounds_, self.upper_bounds_, axis=1)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["sunshine"] = np.log1p(X["sunshine"].clip(lower=0))
        X["humidity_pressure_interaction"] = X["humidity"] * X["pressure"]
        return X

numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove("rainfall")

preprocessor = ColumnTransformer([
    ("num_scaler", StandardScaler(), numeric_cols)
])

def create_pipeline(model):
    return Pipeline([
        ("outlier_cap", OutlierCapTransformer()),
        ("feature_eng", FeatureEngineer()),
        ("scaler", preprocessor),
        ("model", model)
    ])

X = df_train.drop(columns=["rainfall"])
y = df_train["rainfall"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)



# ANALYZE - Optimize ExtraTreesClassifier with Optuna
def objective_etc(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "random_state": 42,
        "class_weight": "balanced",
    }
    model = ExtraTreesClassifier(**params)
    pipeline = create_pipeline(model)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)

study_etc = optuna.create_study(direction="maximize")
study_etc.optimize(objective_etc, n_trials=50)
best_etc_params = study_etc.best_params



# VISUALIZE - Model Performance
et_final = ExtraTreesClassifier(**best_etc_params)
pipeline_et = create_pipeline(et_final)
pipeline_et.fit(X_train, y_train)

pred_et = pipeline_et.predict_proba(X_val)[:, 1]
auc_et = roc_auc_score(y_val, pred_et)

fpr_et, tpr_et, _ = roc_curve(y_val, pred_et)
plt.figure(figsize=(7, 5))
plt.plot(fpr_et, tpr_et, label=f"ExtraTrees (AUC={auc_et:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.title("Validation ROC")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()


# ACT - Make Predictions & Save Submission
test_pred_et = pipeline_et.predict_proba(df_test)[:, 1]
submission = pd.DataFrame({"id": df_test.index, "rainfall": test_pred_et})
submission.to_csv("submission.csv", index=False)
print("Created submission.csv!")


submission

