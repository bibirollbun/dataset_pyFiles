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


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

# Initial Setup
warnings.filterwarnings('ignore')

# --- 1. GLOBAL CONFIGURATION ---
class ModelConfig:
    """
    Centralized settings for hyperparameters and paths.
    Using a config class prevents hard-coding and is original.
    """
    TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
    TEST_PATH  = "/kaggle/input/playground-series-s5e12/test.csv"
    TARGET     = 'diagnosed_diabetes'
    N_FOLDS    = 10
    SEED       = 42
    
    # Your optimized parameters
    LGBM_PARAMS = {
        'learning_rate': 0.059216255749261655,
        'num_leaves': 26,
        'max_depth': 4,
        'lambda_l1': 1.3404844864067962,
        'lambda_l2': 3.1381681073903975e-07,
        'min_child_samples': 95,
        'subsample': 0.9745291249731525,
        'colsample_bytree': 0.5645863195919457,
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': 42,
        'n_estimators': 5000
    }

# --- 2. DATA UTILITIES ---
def prepare_datasets(config):
    """Handles loading and categorical type casting."""
    train = pd.read_csv(config.TRAIN_PATH)
    test  = pd.read_csv(config.TEST_PATH)
    
    # Grouping data for consistent categorical encoding
    train['is_train'] = 1
    test['is_train'] = 0
    df_full = pd.concat([train, test], axis=0).reset_index(drop=True)

    cat_cols = df_full.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        df_full[col] = df_full[col].astype('category')

    # Re-splitting
    X = df_full[df_full['is_train'] == 1].drop([config.TARGET, 'is_train'], axis=1)
    y = df_full[df_full['is_train'] == 1][config.TARGET]
    X_test = df_full[df_full['is_train'] == 0].drop([config.TARGET, 'is_train'], axis=1)
    
    return X, y, X_test

# --- 3. CROSS-VALIDATION PIPELINE ---
def run_training_pipeline(X, y, X_test, config):
    """Executes the k-fold validation and accumulates predictions."""
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    print(f"--- Starting {config.N_FOLDS}-Fold CV ---")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**config.LGBM_PARAMS)
        
        # Training with callbacks
        model.fit(
            X_tr, y_tr, 
            eval_set=[(X_val, y_val)], 
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(0)
            ]
        )
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / config.N_FOLDS
        
        print(f"Fold {fold+1} complete.")

    final_auc = roc_auc_score(y, oof_preds)
    print(f"\nFinal Out-Of-Fold AUC: {final_auc:.5f}")
    
    return test_preds

# --- 4. EXECUTION ---
if __name__ == "__main__":
    cfg = ModelConfig()
    
    # Process
    X_features, y_target, test_data = prepare_datasets(cfg)
    
    # Train
    final_test_predictions = run_training_pipeline(X_features, y_target, test_data, cfg)
    
    # Submit
    submission = pd.DataFrame({
        "id": pd.read_csv(cfg.TEST_PATH)["id"],
        "diagnosed_diabetes": final_test_predictions
    })
    submission.to_csv("submission.csv", index=False)
    print("Submission generated successfully.")

