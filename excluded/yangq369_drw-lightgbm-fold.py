import os
import pickle
import polars as pl
import numpy as np
import pandas as pd
import gc
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer
from pytorch_lightning.loggers import WandbLogger
import lightgbm as lgb
from pandas import read_parquet

from datetime import datetime
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import pearsonr


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def _pearsonr(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

class VotingModel(BaseEstimator, RegressorMixin):
    """
    A voting ensemble model that averages predictions from multiple estimators.

    Parameters:
    - estimators: List of estimators to include in the voting ensemble

    Methods:
    - fit(X, y=None): No training is performed as it's just an aggregator.
    - predict(X): Returns the average prediction from all included estimators.
    - predict_proba(X): Returns the average class probabilities from all included estimators.
    """

    def __init__(self, estimators):
        """
        Initializes the VotingModel with a list of estimators.

        Parameters:
        - estimators: List of estimators to include in the voting ensemble
        """
        super().__init__()
        self.estimators = estimators

    def fit(self, X, y=None):
        """Fits the voting model (no operation)."""
        return self

    def predict(self, X):
        """Returns the average prediction from all included estimators."""
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

    def predict_proba(self, X):
        """Returns the average class probabilities from all included estimators."""
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)
    
def reduce_mem_usage(df):
    """
    Optimizes the memory usage of a DataFrame by downcasting numeric columns to smaller data types.

    Parameters:
    - df: DataFrame to be optimized

    Returns:
    - df: Optimized DataFrame
    """

    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type) == "category":
            continue

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            continue
    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def feature_engineering(df):
    # Create interaction features for df
    df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
    df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
    df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
    df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']
    df['buy_sell_interaction'] = df['buy_qty'] * df['sell_qty']

    # Calculate spread indicators for df
    df['spread_indicator'] = (df['ask_qty'] - df['bid_qty']) / (df['ask_qty'] + df['bid_qty'] + 1e-8)

    # Volume-weighted features for df
    df['volume_weighted_buy'] = df['buy_qty'] * df['volume']
    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['volume_weighted_bid'] = df['bid_qty'] * df['volume']
    df['volume_weighted_ask'] = df['ask_qty'] * df['volume']

    # NEW FEATURES - Add ratio features
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    df['bid_ask_ratio'] = df['bid_qty'] / (df['ask_qty'] + 1e-8)

    # NEW FEATURES - Add order flow imbalance
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)

    # NEW FEATURES - Add market pressure indicators
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-8)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-8)

    # ADDITIONAL NEW MARKET FEATURES - Liquidity measures
    df['total_liquidity'] = df['bid_qty'] + df['ask_qty']
    df['liquidity_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_liquidity'] + 1e-8)
    df['relative_spread'] = (df['ask_qty'] - df['bid_qty']) / (df['volume'] + 1e-8)

    # ADDITIONAL NEW MARKET FEATURES - Trade intensity
    df['trade_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-8)
    df['avg_trade_size'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-8)
    df['net_trade_flow'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-8)

    # ADDITIONAL NEW MARKET FEATURES - Market depth and activity
    df['depth_ratio'] = df['total_liquidity'] / (df['volume'] + 1e-8)
    df['volume_participation'] = (df['buy_qty'] + df['sell_qty']) / (df['total_liquidity'] + 1e-8)
    df['market_activity'] = df['volume'] * df['total_liquidity']

    # ADDITIONAL NEW MARKET FEATURES - Execution quality indicators
    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
    df['realized_volatility_proxy'] = np.abs(df['order_flow_imbalance']) * df['volume']

    # ADDITIONAL NEW MARKET FEATURES - Normalized volumes
    df['normalized_buy_volume'] = df['buy_qty'] / (df['bid_qty'] + 1e-8)
    df['normalized_sell_volume'] = df['sell_qty'] / (df['ask_qty'] + 1e-8)

    # ADDITIONAL NEW MARKET FEATURES - Complex interactions
    df['liquidity_adjusted_imbalance'] = df['order_flow_imbalance'] * df['depth_ratio']
    df['pressure_spread_interaction'] = df['buying_pressure'] * df['spread_indicator']

    # Replace any inf or -inf values with NaN, then fill NaN with 0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df 


%%time
path = r'/kaggle/input/drw-crypto-market-prediction/'
train = reduce_mem_usage(read_parquet(path+r'train.parquet'))
test = reduce_mem_usage(read_parquet(path+r'test.parquet'))


%%time
train = feature_engineering(train)
test = feature_engineering(test)


x_feaures = [f"X{num}" for num in range(1,890)]


import random
random.seed(42)
x_feaures = random.sample(train.columns[:-2].tolist(),100)


feature_names = [
                 'X863',
                 'X856',
                 'X344',
                 'X598',
                 'X862',
                 'X385',
                 'X852',
                 'X603',
                 'X860',
                 'X674',
                 'X415',
                 'X345',
                 'X137',
                 'X855',
                 'X174',
                 'X302',
                 'X178',
                 'X532',
                 'X168',
                 'X612',
                 'bid_qty',
                 'ask_qty',
                 'buy_qty',
                 'sell_qty',
                 'volume',
                 'bid_ask_interaction',
                 'bid_buy_interaction',
                 'bid_sell_interaction',
                 'ask_buy_interaction',
                 'ask_sell_interaction'
                        ]
# feature_names = x_feaures
# feature_names = feature_names + new_features
label_name = 'label'


train.loc['2023-03-01 00:00:00':'2023-05-01 00:00:00','Fold'] =  1
train.loc['2023-05-01 00:00:00':'2023-07-01 00:00:00','Fold'] =  2
train.loc['2023-07-01 00:00:00':'2023-09-01 00:00:00','Fold'] =  3
train.loc['2023-09-01 00:00:00':'2023-11-01 00:00:00','Fold'] =  4
train.loc['2023-11-01 00:00:00':'2024-01-01 00:00:00','Fold'] =  5
train.loc['2024-01-01 00:00:00':'2024-03-01 00:00:00','Fold'] =  6


def create_time_weights(n_samples, decay_factor=0.95):
    """
    Create exponentially decaying weights based on sample position.
    More recent samples (higher indices) get higher weights.
    decay_factor controls the rate of decay (0.95 = 5% decay per time unit)
    """
    positions = np.arange(n_samples)
    # Normalize positions to [0, 1] range
    normalized_positions = positions / (n_samples - 1)
    # Apply exponential weighting
    weights = decay_factor ** (1 - normalized_positions)
    # Normalize weights to sum to n_samples (maintains scale)
    weights = weights * n_samples / weights.sum()
    return weights


train['weight'] = create_time_weights(len(train), decay_factor=0.95)


def pearsonr_coeff(preds, data):
    y_true = data.get_label()
    # weights = data.get_weight()
    valid_score = _pearsonr(y_true, preds)
    return 'pearsonr_coeff_score',valid_score,True


 # 训练模型
def TrainModel(train_data, valid_data, lgb_params):
    print("Training Model...")
    model = lgb.train(lgb_params,
                        train_data,
                        num_boost_round=150,
                        valid_sets=[valid_data],
                        feval=pearsonr_coeff,
                        callbacks=[
                        # lgb.callback.early_stopping(stopping_rounds=300),
                        lgb.callback.log_evaluation(period=50)]
                        )

    valid_pred = model.predict(valid_data.get_data())
    valid_score = _pearsonr(valid_data.get_label(),valid_pred)
    print("Valid Score:", valid_score)
    return model,valid_score


import xgboost as xgb
models = []
valid_scores = []
for fold in range(1,6):
    X_train = train[(train['Fold']!=fold)][ feature_names ]
    w_train = train[(train['Fold']!=fold)][ 'weight' ]
    X_valid = train[train['Fold']==6][ feature_names ]
    w_valid = train[train['Fold']==6][ 'weight' ]
    y_train = train[(train['Fold']!=fold)][ label_name ]
    y_valid = train[train['Fold']==6][ label_name ]


    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train,free_raw_data=False).construct()
    valid_data = lgb.Dataset(X_valid, label=y_valid, weight=w_valid,reference=train_data, free_raw_data=False).construct()
    print(f'train time {X_train.index.min()},{X_train.index.max()}')
    print(f'valid time {X_valid.index.min()},{X_valid.index.max()}')

    lgb_params = {
            "boosting_type": "gbdt",
            "objective": "regression",       # 回归任务
            "metric": "mae",                 # 使用 MAE 作为评估指标
            "colsample_bytree": 0.55,
            "learning_rate": 0.021,
            "min_child_samples": 32,
            "min_child_weight": 0.15,
            'max_depth':-1,
            "n_jobs": -1,
            "num_leaves":64,
            "random_state": 42,
            "reg_alpha": 80,
            "reg_lambda": 100,
            "subsample": 0.85,
            "verbosity": 1,  
            "device": "gpu",                 # 使用 GPU 加速
            # "max_bin":1024
            }

    model,valid_score = TrainModel(train_data,valid_data,lgb_params)

    models.append(model)
    valid_scores.append(valid_score)
print(f'Average score is {np.mean(valid_scores)}')


lgbm = VotingModel(models)
submission = pd.read_csv(path+r'sample_submission.csv')
submission['prediction'] = lgbm.predict(test[feature_names])
submission.to_csv(r'submission.csv',index=False)













