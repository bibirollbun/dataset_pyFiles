"""
Bank Term Deposit Prediction - Enhanced LightGBM Model
Inspired by "Bank Term Deposit: Single LightGBM" by bizen250
Enhanced with conservative feature engineering and hyperparameter tuning
"""

import pandas as pd
import numpy as np

from typing import Tuple
import random
import time
import os
import gc

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from tqdm.auto import tqdm

import warnings
warnings.simplefilter('ignore')

tqdm.pandas()

%matplotlib inline

# Configuration - conservative improvements
class CFG:
    mode = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'localhost')
    path = "../input/playground-series-s5e8/"
    original = "../input/bank-marketing-dataset-full/"

    n_splits = 5 
    seed = 42
    
    learning_rate = 0.08  # Slightly reduced for better convergence
    num_boost_round = 40000  # Increased number of iterations
    early_stopping_rounds = 150  # Slightly increased patience
    verbose_eval = False if mode=='Batch' else 200 if learning_rate>=1e-1 else 500

    target = "y"
    plot_importance = False

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG.seed)

# Load data
train = pd.read_csv(CFG.path + "train.csv").drop(columns=['id'])
test = pd.read_csv(CFG.path + "test.csv").drop(columns=['id'])

original = pd.read_csv(CFG.original + "bank-full.csv", sep=";")
original[CFG.target] = (original[CFG.target]=="yes").astype(int)

features = test.columns.to_list()

# Minimal feature engineering - only the most effective features
def add_simple_features(df):
    df = df.copy()
    
    # Only the most important new features
    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['has_previous'] = (df['previous'] > 0).astype(int)
    df['duration_long'] = (df['duration'] > 300).astype(int)
    df['campaign_multiple'] = (df['campaign'] > 2).astype(int)
    
    # Simple numerical transformations
    df['log_duration'] = np.log1p(df['duration'])
    df['sqrt_age'] = np.sqrt(df['age'])
    
    return df

# Apply minimal enhancements
train_enhanced = add_simple_features(train)
test_enhanced = add_simple_features(test)

# Update feature list
new_features = ['balance_positive', 'has_previous', 'duration_long', 'campaign_multiple', 'log_duration', 'sqrt_age']
all_features = features + new_features

print(f"Original features: {len(features)}")
print(f"New features: {len(new_features)}")
print(f"Total features: {len(all_features)}")

# Slightly improved LightGBM parameters
params = {
    'objective': "binary",
    'metric': 'binary_logloss',
    'categorical_feature': features,  # Only original categorical features
    'verbosity': -1,
    'boosting_type': "gbdt",
    'random_state': CFG.seed,
    'learning_rate': CFG.learning_rate,
    'max_depth': 9,  # Slightly increased
    'num_leaves': 100,  # Slightly increased
    'max_bin': 255,  # Increased for better splits
    'subsample': 0.82,  # Slightly modified
    'colsample_bytree': 0.65,  # Slightly modified
    'subsample_freq': 1,
    'reg_alpha': 2.8,  # Slightly modified regularization
    'reg_lambda': 1.8,
    'min_child_samples': 25,  # Added for stability
    'min_split_gain': 0.005,  # Added to prevent overfitting
    'n_jobs': -1,
    'extra_trees': True,
    'bagging_seed': CFG.seed,
    'feature_fraction_seed': CFG.seed,
}

oof = np.zeros(train.shape[0])
pred = np.zeros(test.shape[0])

cv = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
splitter = cv.split(train_enhanced, train_enhanced[CFG.target])

for fold, (trn_idx, val_idx) in enumerate(splitter):
    start_time = time.time()

    # Apply same transformations to original data
    original_enhanced = add_simple_features(original)
    
    M_train = pd.concat([train_enhanced.iloc[trn_idx], original_enhanced])
    M_train = M_train.drop_duplicates(subset=features, keep="first", ignore_index=True)
    X_train = M_train[all_features]
    y_train = M_train[CFG.target]

    X_valid = train_enhanced.loc[val_idx, all_features]
    y_valid = train_enhanced.loc[val_idx, CFG.target]
    X_test = test_enhanced[all_features].copy()
    
    # Categorical features only for original features
    X_train[features] = X_train[features].astype("category")
    X_valid[features] = X_valid[features].astype("category")
    X_test[features] = X_test[features].astype("category")

    # Create DMatrix for LightGBM
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    # LightGBM callbacks
    ES = lgb.callback.early_stopping(
        stopping_rounds=CFG.early_stopping_rounds,
        verbose=False
    )
    LE = lgb.log_evaluation(
        period=CFG.verbose_eval,
        show_stdv=True
    )

    # Train the model with early stopping
    model = lgb.train(params, 
                      train_set=dtrain,
                      valid_sets=[dtrain, dvalid],
                      valid_names=["train", "valid"],
                      num_boost_round=CFG.num_boost_round,
                      callbacks=[ES, LE])

    # Evaluate on validation set
    oof[val_idx] = model.predict(X_valid)
    
    # Generate predictions for test set
    pred += model.predict(X_test) / CFG.n_splits

    score = roc_auc_score(y_valid, oof[val_idx])

    end_time = time.time()
    print("----------------------------------------------------------------")
    print(f"fold: {fold:02d}, auc: {score:.6f}, best iteration: {model.best_iteration}, best score: {model.best_score['valid']['binary_logloss']: .6f}, elapsed time: {end_time-start_time: .2f} sec.\n")

    if CFG.plot_importance:
        _, ax = plt.subplots(figsize=(12, 4))
        lgb.plot_importance(model,
                            ax=ax,
                            max_num_features=15,
                            importance_type='gain')
        plt.show()
    
    # Free memory
    del model, dtrain, dvalid
    gc.collect()

score = roc_auc_score(train[CFG.target], oof)
print("----------------------------------------------------------------")
print(f"          auc: {score:.6f}")

# Save results
np.save("oof_conservative.npy", oof)
np.save("pred_conservative.npy", pred)

# Submission
submission = pd.read_csv(CFG.path + "sample_submission.csv")
submission[CFG.target] = pred

submission.to_csv("submission.csv", index=False)
print(f"\nSubmission shape: {submission.shape}")
print(f"Prediction stats - Min: {pred.min():.6f}, Max: {pred.max():.6f}, Mean: {pred.mean():.6f}")

submission

