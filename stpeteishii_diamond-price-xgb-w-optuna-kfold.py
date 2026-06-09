import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
from time import time
from tqdm import tqdm
import optuna
import category_encoders as ce
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import yaml

# --- Data Loading ---
# Load training data
train0 = pd.read_csv("/kaggle/input/predicting-the-price-of-diamond/train.csv")
test0 = pd.read_csv("/kaggle/input/predicting-the-price-of-diamond/test.csv")
submit = pd.read_csv('/kaggle/input/predicting-the-price-of-diamond/submission.csv')
data0 = pd.concat([train0, test0], axis=0)
columns = test0.columns.tolist()
target = ['price']

def labelencoder(df):
    """Encodes object columns using LabelEncoder."""
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
    return df

data = labelencoder(data0)
train = data.iloc[0:len(train0)]
test = data.iloc[len(train0):]
trainY = train[target]
trainX = train.drop(target, axis=1)
testX = test
train_df = trainX
test_df = testX

def create_numeric_feature(input_df):
    """Returns a copy of the numeric columns."""
    use_columns = columns
    return input_df[use_columns].copy()

# --- Timer Class and Feature Creation Function ---
class Timer:
    """A timer class for measuring execution time."""
    def __init__(self, logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None, sep=' '):
        if prefix: format_str = str(prefix) + sep + format_str
        if suffix: format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)

def to_feature(input_df):
    """Applies feature processors to the input DataFrame."""
    processors = [
        create_numeric_feature,
    ]
    
    out_df = pd.DataFrame()
    
    for func in tqdm(processors, total=len(processors)):
        with Timer(prefix='create' + func.__name__ + ' '):
            _df = func(input_df)

        assert len(_df) == len(input_df), func.__name__
        out_df = pd.concat([out_df, _df], axis=1)
        
    return out_df

train_feat_df = to_feature(train_df)
test_feat_df = to_feature(test_df)

# --- Optuna Hyperparameter Tuning for XGBoost ---
# Prepare data structures to save OOF predictions
oof_predictions = {target_name: np.zeros(len(trainX)) for target_name in target}
test_predictions = {target_name: np.zeros(len(testX)) for target_name in target}

def objective(trial, data=trainX, target_col=None):
    """Enhanced objective function with both RMSE and R² tracking for XGBoost"""
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    rmse_scores = []
    r2_scores = []  # Track R² as well
    
    param = {
        'lambda': trial.suggest_float('lambda', 0.001, 0.002, log=True),
        'alpha': trial.suggest_float('alpha', 6, 12),
        'subsample': trial.suggest_float('subsample', 0.56, 0.57),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 0.76),
        'learning_rate': trial.suggest_float('learning_rate', 0.06, 0.08),
        'verbosity': trial.suggest_int('verbosity', 0, 0),  # XGBoost uses 0-3 for verbosity
        'random_state': trial.suggest_int('random_state', 60, 68),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'n_estimators': 1000,
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 15),
        'gamma': trial.suggest_float('gamma', 0, 0.1),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
    }
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(data, target_col)):
        X_tr, X_val = data.iloc[trn_idx], data.iloc[val_idx]
        y_tr, y_val = target_col.iloc[trn_idx], target_col.iloc[val_idx]
        
        model = xgb.XGBRegressor(**param)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=False
        )
        
        val_preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, val_preds, squared=False)
        r2 = r2_score(y_val, val_preds)  # Calculate R²
        
        rmse_scores.append(rmse)
        r2_scores.append(r2)
    
    # Store both metrics for analysis
    trial.set_user_attr('mean_r2', np.mean(r2_scores))
    trial.set_user_attr('std_r2', np.std(r2_scores))
    
    return np.mean(rmse_scores)

BestTrials = []
Studies = []

for i in range(1):
    target_col = trainY.iloc[:, i]
    target_name = target[i]
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, trainX[columns], target_col), n_trials=300)
    
    print(f'Target: {target_name}')
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)

    Best_trial = study.best_trial.params
    with open(f"Best_trial{i}.yaml", "w") as yaml_file:
        yaml.dump(Best_trial, yaml_file)

    # XGBoost specific fixed parameters
    fix_dict = {
        'objective': 'reg:squarederror',
        'n_estimators': 1000,
        'eval_metric': 'rmse',
        'verbosity': 0,
    }
    Best_trial.update(fix_dict)
    BestTrials.append(Best_trial)
    Studies.append(study)

# --- Visualization ---
import optuna.visualization.matplotlib as vis_matplotlib

for i, study in enumerate(Studies):
    print()
    print('================' * 4)
    print(target[i])
    display(vis_matplotlib.plot_optimization_history(study))
    display(vis_matplotlib.plot_slice(study))
    display(vis_matplotlib.plot_param_importances(study))
    print()

# --- Final Model Training and Prediction with XGBoost ---
train = trainX
target_df = trainY
test = testX
test_preds = np.zeros((len(test), 1))  # Only one target: price

for i in range(1):
    target_col = target_df.iloc[:, i]
    target_name = target[i]
    
    kf = KFold(n_splits=5, random_state=48, shuffle=True)
    fold_test_preds = np.zeros((len(test), kf.n_splits))
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(train[columns], target_col)):
        print(f'Target: {target_name}, Fold: {fold+1}')
        
        X_tr, X_val = train[columns].iloc[trn_idx], train[columns].iloc[val_idx]
        y_tr, y_val = target_col.iloc[trn_idx], target_col.iloc[val_idx]
        
        Best_trial = BestTrials[i]
        model = xgb.XGBRegressor(**Best_trial)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=False
        )
        
        oof_preds = model.predict(X_val)
        oof_predictions[target_name][val_idx] = oof_preds
        
        fold_test_pred = model.predict(test[columns])
        fold_test_preds[:, fold] = fold_test_pred
        
        rmse = mean_squared_error(y_val, oof_preds, squared=False)
        print(f'RMSE: {rmse:.4f}')
    
    test_preds[:, i] = fold_test_preds.mean(axis=1)

# --- Evaluation and Submission ---
print("\nOOF Prediction Evaluation:")
for i, target_name in enumerate(target):
    oof_rmse = mean_squared_error(target_df.iloc[:, i], oof_predictions[target_name], squared=False)
    print(f'{target_name}: OOF RMSE = {oof_rmse:.4f}')

oof_df = pd.DataFrame(oof_predictions)
oof_df.to_csv('oof_predictions_xgb.csv', index=False)
print("OOF predictions saved to oof_predictions_xgb.csv")

# Prepare submission
print(f"\nDebug info:")
print(f"submit shape: {submit.shape}")
print(f"test_preds shape: {test_preds.shape}")
print(f"submit columns: {submit.columns.tolist()}")

# Ensure test_preds has the right shape
if test_preds.ndim == 1:
    test_preds = test_preds.reshape(-1, 1)

# Check if shapes match
if test_preds.shape[1] != (submit.shape[1] - 1):
    print(f"Warning: Shape mismatch! test_preds has {test_preds.shape[1]} columns, but submit expects {submit.shape[1] - 1} prediction columns")
    
    # Adjust test_preds to match submit format
    if test_preds.shape[1] > (submit.shape[1] - 1):
        test_preds = test_preds[:, :(submit.shape[1] - 1)]
        print(f"Truncated test_preds to {test_preds.shape}")
    else:
        # Pad with zeros if needed
        padding_cols = (submit.shape[1] - 1) - test_preds.shape[1]
        padding = np.zeros((test_preds.shape[0], padding_cols))
        test_preds = np.hstack([test_preds, padding])
        print(f"Padded test_preds with {padding_cols} zeros to shape {test_preds.shape}")

# Convert to appropriate dtype to avoid warning
submit.iloc[:, 1:] = test_preds.astype(np.float64)

submit.to_csv('submission_xgb.csv', index=False)
print("XGBoost submission file created and saved.")
display(submit.head())

