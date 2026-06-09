# Upgrade
!pip install -U --q scikit-learn


# --------------------------
# Libraries
# --------------------------


# General
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, string, re, random, gc, pickle, math,warnings
import json, glob
from itertools import *
from datetime import date
from tqdm.keras import TqdmCallback
from tqdm import tqdm
from pathlib import Path

# Sklearn
from sklearn.model_selection import *
from sklearn.feature_extraction import *
from sklearn.metrics import *
from sklearn.metrics import pairwise
from sklearn.preprocessing import *
from sklearn.utils import *
from sklearn.pipeline import *
from sklearn.compose import *
from sklearn.ensemble import HistGradientBoostingClassifier

# Stats
import scipy
from scipy.stats import *
from scipy.sparse import csr_matrix

# Setting
pd.set_option('max_colwidth',None)
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

data_path = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith('csv'):
            data_path.append(os.path.join(dirname, filename))


# --------------------------
# Config
# --------------------------

class Config:
    RANDOM_STATE = 210
    VAL_FRAC = 0.2
    N_SPLITS = 5
    N_ITER = 25

config = Config()


# --------------------------
# Data
# --------------------------

# Load
raw_train = pd.read_csv(data_path[1])
raw_test = pd.read_csv(data_path[2])
sub = pd.read_csv(data_path[0])

# Drop 
remove = ['id']
df_train = raw_train.drop(remove, axis=1)
df_test = raw_test.drop(remove, axis=1)

# Basic stats & view
print(f"Train: {df_train.shape} | Test: {df_test.shape}\n")
df_train.head()



# --------------------------
# Data Definition
# --------------------------

# features & target
y = df_train['diagnosed_diabetes'].values
X = df_train.drop(columns=['diagnosed_diabetes'])


num_cols = ['age','alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 'diet_score',
            'sleep_hours_per_day','screen_time_hours_per_day','bmi','waist_to_hip_ratio','systolic_bp',
            'diastolic_bp','heart_rate','cholesterol_total','hdl_cholesterol','ldl_cholesterol',
            'triglycerides']
 
cat_cols = ['gender','ethnicity','education_level', 'income_level','smoking_status','employment_status']
bin_cols = ['family_history_diabetes','hypertension_history','cardiovascular_history','diagnosed_diabetes']


# ------------------------------------------------------------
# Build preprocessing and baseline model pipeline
# ------------------------------------------------------------

def make_pipeline(num_cols, bin_cols, cat_cols):
    """
    Preprocess:
      - Numeric: median imputation
      - Binary 0/1: most_frequent imputation (keep as numeric; no one-hot)
      - Categorical: most_frequent imputation + one-hot (handle_unknown)
    Model: HistGradientBoostingClassifier with reasonable defaults
    """
    preprocess = ColumnTransformer(
        transformers=[('cat', Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)],
        remainder='drop',
        verbose_feature_names_out=False
    )

    clf = HistGradientBoostingClassifier(
        loss='log_loss',
        learning_rate=0.05,
        max_iter=300,                 # higher than default for a stronger baseline
        max_leaf_nodes=31,
        min_samples_leaf=20,
        max_bins=255,
        l2_regularization=0.0,
        early_stopping=True,
        validation_fraction=config.VAL_FRAC,
        n_iter_no_change=10,
        class_weight='balanced',      # helpful if classes are imbalanced
        random_state=config.RANDOM_STATE
    )

    pipe = Pipeline(steps=[
        ('prep', preprocess),
        ('clf', clf)
    ])
    return pipe


# ------------------------------------------------------------
# Cross-validated baseline ROC-AUC (and PR-AUC)
# ------------------------------------------------------------

def evaluate_baseline(pipe, X, y):
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_validate(
        pipe, X, y, cv=cv,
        scoring={'roc_auc': 'roc_auc', 'ap': 'average_precision'},
        return_train_score=False,
        n_jobs=-1
    )
    print("=== Baseline (HistGradientBoostingClassifier) ===")
    print(f"ROC-AUC (mean ± std): {scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}")
    print(f"Avg Precision (PR-AUC) mean: {scores['test_ap'].mean():.4f}")
    return cv, scores


# ------------------------------------------------------------
# Hyperparameter optimization for ROC-AUC
# ------------------------------------------------------------

def make_search(pipe, cv):
    # Parameter distributions (prefixed 'clf__' for pipeline)
    param_distributions = {
        'clf__max_iter': randint(200, 1200),                 # more rounds if data is larger
        'clf__max_leaf_nodes': randint(15, 255),             # leaf-wise complexity
        'clf__min_samples_leaf': randint(1, 50),
        'clf__learning_rate': loguniform(1e-3, 2e-1),        # log-scaled LR
        'clf__max_bins': randint(64, 255),                   # histogram bins
        'clf__l2_regularization': loguniform(1e-10, 1e-2),   # log-scaled L2
        'clf__max_depth': [None, 3, 4, 5, 6, 8, 10]          # optional cap on depth
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=config.N_ITER,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        verbose=1,
        refit=True
    )
    return search


def optimize_and_report(search, X, y):
    search.fit(X, y)

    print("\n=== Hyperparameter Search Results ===")
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print("Best Params:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    # Re-evaluate with a new CV seed to estimate generalization
    tuned_cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    tuned_scores = cross_validate(
        search.best_estimator_, X, y, cv=tuned_cv,
        scoring={'roc_auc': 'roc_auc', 'ap': 'average_precision'},
        return_train_score=False,
        n_jobs=-1
    )
    print("\n=== Tuned Model Re-evaluation ===")
    print(f"Tuned ROC-AUC (mean ± std): {tuned_scores['test_roc_auc'].mean():.4f} ± {tuned_scores['test_roc_auc'].std():.4f}")
    print(f"Tuned Avg Precision (PR-AUC) mean: {tuned_scores['test_ap'].mean():.4f}")

    return search.best_estimator_, tuned_scores


# ------------------------------------------------------------
# Orchestrate
# ------------------------------------------------------------

if __name__ == '__main__':
    pipe = make_pipeline(num_cols, bin_cols, cat_cols)
    cv, baseline_scores = evaluate_baseline(pipe, X, y)

    search = make_search(pipe, cv)
    best_model, tuned_scores = optimize_and_report(search, X, y)

    # save_model(best_model, path='best_hgb_pipeline.joblib')


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
y_pred = best_model.predict(df_test)


# ------------------------------------------------------------
# Submission file
# ------------------------------------------------------------
sub['diagnosed_diabetes'] = y_pred.tolist()
sub.to_csv('submission.csv', index=False)

