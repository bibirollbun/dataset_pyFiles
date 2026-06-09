import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from scipy.stats import pearsonr


train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')


train.head(2)


x_features = [col for col in train.columns if col.startswith('X')]

key_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']

label_col = 'label'


len(x_features)


def val_loss_function(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # if np.std(y_true) == 0 or np.std(y_pred) == 0:
    #     return 1.0 

    corr, _ = pearsonr(y_true, y_pred)
    return corr

def cross_val_predict(model, X_train, y_train, X_test, val_loss_function, n_splits=5, random_state=42):
    print(f"Model: {model.__class__.__name__}")

    oof_preds = np.zeros(X_train.shape[0])
    test_preds = np.zeros(X_test.shape[0])

    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    val_score = 0
    val_score_log = 0
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Fold {fold + 1}")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        cur_val_score = val_loss_function(y_val, val_preds)
        print(f"Current validation score: {cur_val_score}")
          
        val_score += cur_val_score / n_splits

        test_preds += model.predict(X_test) / n_splits


    print(f"Average validation score: {val_score}")
    return oof_preds, test_preds, val_score


X_train = train[key_features]
X_test = test[key_features]
y_train = train[label_col]


models = [
    LGBMRegressor(
        boosting_type='gbdt',
        device='gpu'  
    ),
    XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        tree_method='gpu_hist', 
        predictor='gpu_predictor'
    ),
    CatBoostRegressor(
        verbose=0,
        task_type='GPU',  
        devices='0'       
    )
]

results = {}

for model in models:
    oof, test, score = cross_val_predict(model, X_train, y_train, X_test, val_loss_function)
    results[model.__class__.__name__] = {
        "oof": oof,
        "test": test,
        "score": score
    }
    print(f"Final validation score for {model.__class__.__name__}: {score}\n")


y_pred = (results['LGBMRegressor']['test'])


sub = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
sub['prediction'] = y_pred
sub.to_csv('submission.csv', index = False)
sub

