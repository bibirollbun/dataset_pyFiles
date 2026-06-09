import warnings
warnings.filterwarnings("ignore")

# === Imports ===
import numpy as np
import pandas as pd
import polars as pl
import os
from functools import partial
import scipy as sp
import sys
import optuna

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import cohen_kappa_score
import xgboost as xgb
import random

# === Constants ===
SEED = 42
NUM_FOLDS = 10
TRAIN_PATH = '/kaggle/input/playground-series-s3e5/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s3e5/test.csv'
SAMPLE_SUBMISSION = '/kaggle/input/playground-series-s3e5/sample_submission.csv'

# === Reproducibility ===
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything()

# === Optimized Rounder (Faster Version) ===
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = [3.5, 4.5, 5.5, 6.5, 7.5]

    def _kappa_loss(self, coef, X, y):
        X_p = np.digitize(X, bins=np.sort(coef)) + 3
        return -cohen_kappa_score(y, X_p, weights='quadratic')

    def fit(self, X, y):
        loss_partial = lambda coef: self._kappa_loss(coef, X, y)
        result = sp.optimize.minimize(
            loss_partial, 
            self.coef_, 
            method='nelder-mead',
            options={'maxiter': 100}  # Reduced iterations
        )
        self.coef_ = np.sort(result.x)

    def predict(self, X, coef):
        return np.digitize(X, bins=np.sort(coef)) + 3

    def coefficients(self):
        return self.coef_

# === Feature Engineering (Optimized) ===
def feature_engineering(df):
    df = df.copy()
    
    # Basic transformations - using vectorized operations
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Log and sqrt transformations in one go
    df = pd.concat([
        df,
        pd.DataFrame({
            f'{col}_log': np.log1p(df[col].clip(lower=0))
            for col in numeric_cols
        }),
        pd.DataFrame({
            f'{col}_sqrt': np.sqrt(df[col].clip(lower=0))
            for col in numeric_cols
        })
    ], axis=1)
    
    # Important columns for interactions
    important_cols = ['alcohol', 'volatile_acidity', 'sulphates', 'total_sulfur_dioxide', 
                     'fixed_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide']
    important_cols = [col for col in important_cols if col in df.columns]
    
    # Create polynomial features for important columns
    for col in important_cols:
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_cubed'] = df[col] ** 3
    
    # Create interaction features for important columns
    for i in range(len(important_cols)):
        for j in range(i+1, len(important_cols)):
            col1, col2 = important_cols[i], important_cols[j]
            df[f'{col1}_{col2}_interact'] = df[col1] * df[col2]
            df[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e-6)
    
    # Create some domain-specific features
    if 'alcohol' in df.columns and 'volatile_acidity' in df.columns:
        df['alcohol_va_ratio'] = df['alcohol'] / (df['volatile_acidity'] + 1e-6)
    
    if 'sulphates' in df.columns and 'chlorides' in df.columns:
        df['sulphates_chlorides_ratio'] = df['sulphates'] / (df['chlorides'] + 1e-6)
    
    if 'total_sulfur_dioxide' in df.columns and 'free_sulfur_dioxide' in df.columns:
        df['bound_sulfur_dioxide'] = df['total_sulfur_dioxide'] - df['free_sulfur_dioxide']
    
    return df

# === Objective Function for Optuna ===
def optuna_objective(trial, X, y):
    params = {
        'objective': 'reg:squarederror',
        'random_state': SEED,
        'nthread': -1,
        'tree_method': 'hist',
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 10.0),
        'eta': trial.suggest_float('eta', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'max_delta_step': trial.suggest_float('max_delta_step', 0, 100),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0),
    }

    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=SEED)
    kappa_scores = []

    for train_idx, val_idx in rkf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(params, dtrain, num_boost_round=2000,
                          evals=[(dval, 'val')],
                          early_stopping_rounds=100, verbose_eval=False)

        preds = model.predict(dval)
        preds = 3 + (preds - preds.min()) * 5 / (preds.max() - preds.min())

        optR = OptimizedRounder()
        optR.fit(preds, y_val)
        pred_labels = optR.predict(preds, optR.coefficients())

        kappa = cohen_kappa_score(y_val, pred_labels, weights='quadratic')
        kappa_scores.append(kappa)

    return np.mean(kappa_scores)

# === Cross Validation Function (Optimized) ===
def cross_valid(params, X, y, X_test):
    rkf = RepeatedStratifiedKFold(n_splits=NUM_FOLDS, n_repeats=3, random_state=SEED)

    train_oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []
    optR = OptimizedRounder()

    for fold, (train_idx, val_idx) in enumerate(rkf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)

        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=2000,
            evals=[(dval, 'val')],
            early_stopping_rounds=100,
            verbose_eval=False
        )

        val_preds = model.predict(dval)
        val_preds = 3 + (val_preds - val_preds.min()) * 5 / (val_preds.max() - val_preds.min())

        optR.fit(val_preds, y_val)
        val_preds_rounded = optR.predict(val_preds, optR.coefficients())

        train_oof[val_idx] = val_preds_rounded

        test_fold_preds = model.predict(dtest)
        test_fold_preds = 3 + (test_fold_preds - test_fold_preds.min()) * 5 / (test_fold_preds.max() - test_fold_preds.min())
        test_fold_preds = optR.predict(test_fold_preds, optR.coefficients())

        test_preds += test_fold_preds / (NUM_FOLDS * 3)

        score = cohen_kappa_score(y_val, val_preds_rounded, weights='quadratic')
        scores.append(score)
        print(f"Fold {fold} - Kappa: {score:.4f}")

    print(f"Mean Kappa: {np.mean(scores):.4f}")
    print(f"OOF Kappa: {cohen_kappa_score(y, train_oof, weights='quadratic'):.4f}")

    return train_oof, test_preds

# === Main (Optimized) ===
def main():
    # Load data with Polars (faster on M1), convert to Pandas for modeling
    train_df = pl.read_csv(TRAIN_PATH).to_pandas()
    test_df = pl.read_csv(TEST_PATH).to_pandas()
    sample_submission = pl.read_csv(SAMPLE_SUBMISSION).to_pandas()

    y = train_df['quality']
    X = feature_engineering(train_df.drop(columns=['Id', 'quality']))
    X_test = feature_engineering(test_df.drop(columns=['Id']))

    print("Starting Optuna optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: optuna_objective(trial, X, y), n_trials=30)  # Increased trials
    best_params = study.best_params
    best_params['objective'] = 'reg:squarederror'
    best_params['random_state'] = SEED
    best_params['nthread'] = -1
    best_params['tree_method'] = 'hist'

    print("Best parameters:", best_params)
    train_oof, test_preds = cross_valid(best_params, X, y, X_test)

    sample_submission['quality'] = test_preds.astype(int)
    sample_submission.to_csv("submission4.csv", index=False)
    print("submission4.csv created successfully!")

if __name__ == '__main__':
    main()

