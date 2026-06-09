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


import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score
import lightgbm as lgb
import optuna


# Preprocessing functions
def preprocess(df):
    num_cols = ['temparature', 'humidity']
    cat_cols = ['winddirection']
    
    # Chain assignment (one-time fill)
    medians = {col: df[col].median() for col in num_cols}
    modes = {col: df[col].mode()[0] for col in cat_cols}
    df.fillna({**medians, **modes}, inplace=True) 
    
    # Add missing tags
    for col in num_cols + cat_cols:
        df[f'{col}_missing'] = df[col].isnull().astype(int)
    
    # Temporal feature processing
    df['day'] = pd.to_datetime(df['day'])
    df['season'] = df['day'].dt.month % 12 // 3 + 1
    
    return df.infer_objects(copy=False)  # Type inference optimization



def feature_engineering(df):
    if {'maxtemp', 'mintemp'}.issubset(df.columns):
        df['temp_diff'] = df['maxtemp'] - df['mintemp']
    if {'temparature', 'dewpoint'}.issubset(df.columns):
        df['dew_diff'] = df['temparature'] - df['dewpoint']
    window_size = 3
    df['temp_rolling_mean'] = df['temparature'].rolling(window=window_size).mean()
    df['windspeed_bin'] = pd.cut(df['windspeed'], bins=3, labels=['low', 'medium', 'high'])
    df['pressure_log'] = np.log1p(df['pressure'])
    return df


# Time series validation
tscv = TimeSeriesSplit(n_splits=5)

from lightgbm import LGBMClassifier, early_stopping, log_evaluation

def train_lgb(X, y):
    # Time series validation set partitioning
    split_point = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]
    
    # Initialize the model
    model = LGBMClassifier(
        objective='binary',
        n_estimators=1000,
        learning_rate=0.05,
        verbosity=-1,  
        force_row_wise=True 
    )
    
    # Configure a callback function
    callbacks = [
        early_stopping(stopping_rounds=50),
        log_evaluation(period=50)  # Evaluation results are output every 50 rounds
    ]
    
    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks
    )
    return model
    # CV
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train,
                 eval_set=(X_val, y_val),
                 early_stopping_rounds=50,
                 verbose=False)
        
        scores.append(roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))
    
    print(f"CV AUC: {np.mean(scores):.4f}")
    return model


def objective(trial, X, y):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
    }
    
    model = lgb.LGBMClassifier(**params, n_estimators=1000)
    score = []
    for train_idx, val_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx],
                 eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
                 early_stopping_rounds=50,
                 verbose=False)
        score.append(roc_auc_score(y.iloc[val_idx], model.predict_proba(X.iloc[val_idx])[:,1]))
    
    return np.mean(score)


from mlxtend.classifier import StackingCVClassifier
from xgboost import XGBClassifier

def model_stacking(X, y):
    base_models = [
        lgb.LGBMClassifier(),
        XGBClassifier(use_label_encoder=False)
    ]
    
    stack_model = StackingCVClassifier(
        classifiers=base_models,
        meta_classifier=lgb.LGBMClassifier(),
        cv=tscv
    )
    
    stack_model.fit(X, y)
    return stack_model



def optimize_threshold(y_true, y_proba):
    thresholds = np.linspace(0.1, 0.5, 50)
    best_f1 = 0
    best_th = 0
    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th



if __name__ == "__main__":
    # Data loading
    df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
    
    # Pre-treatment pipelines
    df = preprocess(df)
    df = feature_engineering(df)
    
    # Feature/target separation
    X = df.drop(['day', 'rainfall'], axis=1)
    y = df['rainfall']
    X = pd.get_dummies(X)  
    
    # Model training
    model = train_lgb(X, y)
    
    # Validation set evaluation (keep the last 20%)
    split_point = int(len(X) * 0.8)
    X_val, y_val = X.iloc[split_point:], y.iloc[split_point:]
    y_proba = model.predict_proba(X_val)[:,1]
    print(f"Final verification of AUC: {roc_auc_score(y_val, y_proba):.4f}")


    test_processed = preprocess(test_df.copy())
    test_processed = feature_engineering(test_processed)
    
    # Feature Alignment (Key!) Make sure to be consistent with the characteristics of the training set)
    test_features = test_processed.drop(['day'], axis=1) 
    test_features = pd.get_dummies(test_features)
    
    # Align feature columns (0 for features that are populated in the training set but not in the test set)
    train_columns = X.columns
    for col in train_columns:
        if col not in test_features.columns:
            test_features[col] = 0
    
    # Arrange by training set features
    test_features = test_features[train_columns]
    
    # Generate predicted probabilities
    test_pred_proba = model.predict_proba(test_features)[:,1]
    
    # Generate a submission file
    submission = pd.DataFrame({
        'id': test_df['id'].astype(int),  
        'rainfall': test_pred_proba  
    })
    
    submission.to_csv('sub.csv', index=False)
    print("Submission file generated! Preview of the first 5 lines：")
    print(submission.head())

