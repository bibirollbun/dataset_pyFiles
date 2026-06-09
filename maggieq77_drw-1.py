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
    # Original features
    df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
    df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
    df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
    df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']

    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-10)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-10)
    df['log_volume'] = np.log1p(df['volume'])

    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-10)
    
    # === NEW MICROSTRUCTURE FEATURES ===
    
    # Price Pressure Indicators
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-10)
    df['volume_weighted_buy'] = df['buy_qty'] * df['volume']
    
    # Liquidity Depth Measures
    df['total_depth'] = df['bid_qty'] + df['ask_qty']
    df['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['relative_spread'] = np.abs(df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['log_depth'] = np.log1p(df['total_depth'])
    
    # Order Flow Toxicity Proxies
    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['flow_toxicity'] = np.abs(df['order_flow_imbalance']) * df['volume']
    df['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    
    # Market Activity Indicators
    df['volume_depth_ratio'] = df['volume'] / (df['total_depth'] + 1e-10)
    df['activity_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-10)
    df['log_buy_qty'] = np.log1p(df['buy_qty'])
    df['log_sell_qty'] = np.log1p(df['sell_qty'])
    df['log_bid_qty'] = np.log1p(df['bid_qty'])
    df['log_ask_qty'] = np.log1p(df['ask_qty'])
    
    # Microstructure Volatility Proxies
    df['realized_spread_proxy'] = 2 * np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['price_impact_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10)
    df['quote_volatility_proxy'] = np.abs(df['depth_imbalance'])
    
    # Complex Interaction Terms
    df['flow_depth_interaction'] = df['net_order_flow'] * df['total_depth']
    df['imbalance_volume_interaction'] = df['order_flow_imbalance'] * df['volume']
    df['depth_volume_interaction'] = df['total_depth'] * df['volume']
    df['buy_sell_spread'] = np.abs(df['buy_qty'] - df['sell_qty'])
    df['bid_ask_spread'] = np.abs(df['bid_qty'] - df['ask_qty'])
    
    # Information Asymmetry Measures
    df['trade_informativeness'] = df['net_order_flow'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['execution_shortfall_proxy'] = df['buy_sell_spread'] / (df['volume'] + 1e-10)
    df['adverse_selection_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10) * df['volume']
    
    # Market Efficiency Indicators
    df['fill_probability'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['execution_rate'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_efficiency'] = df['volume'] / (df['bid_ask_spread'] + 1e-10)
    
    # Non-linear Transformations
    df['sqrt_volume'] = np.sqrt(df['volume'])
    df['sqrt_depth'] = np.sqrt(df['total_depth'])
    df['volume_squared'] = df['volume'] ** 2
    df['imbalance_squared'] = df['order_flow_imbalance'] ** 2
    
    # Relative Measures
    df['bid_ratio'] = df['bid_qty'] / (df['total_depth'] + 1e-10)
    df['ask_ratio'] = df['ask_qty'] / (df['total_depth'] + 1e-10)
    df['buy_ratio'] = df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['sell_ratio'] = df['sell_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    
    # Market Stress Indicators
    df['liquidity_consumption'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_stress'] = df['volume'] / (df['total_depth'] + 1e-10) * np.abs(df['order_flow_imbalance'])
    df['depth_depletion'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    
    # Directional Indicators
    df['net_buying_ratio'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['directional_volume'] = df['net_order_flow'] * np.log1p(df['volume'])
    df['signed_volume'] = np.sign(df['net_order_flow']) * df['volume']
    
    # Replace infinities and NaNs
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
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
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 设置随机种子

# 假设 train 是你的 DataFrame，选择除最后两列以外的列名
x_features = random.sample(train.columns[:-2].tolist(), 100)



feature_names = [
            "X344","X598","X385","X603",
        "X674","X415","X345","X137","X174","X302",
        "X178","X532","X168","X612",
    'bid_ask_interaction',
    'bid_buy_interaction',
    'bid_sell_interaction',
    'ask_buy_interaction',
    'ask_sell_interaction',
    'volume_weighted_sell',
    'buy_sell_ratio',
    'selling_pressure',
    'log_volume',
    'effective_spread_proxy',
    'bid_ask_imbalance',
    'order_flow_imbalance',
    'liquidity_ratio',
    'net_order_flow',
    'normalized_net_flow',
    'buying_pressure',
    'volume_weighted_buy',
    'total_depth',
    'depth_imbalance',
    'relative_spread',
    'log_depth',
    'kyle_lambda',
    'flow_toxicity',
    'aggressive_flow_ratio',
    'volume_depth_ratio',
    'activity_intensity',
    'log_buy_qty',
    'log_sell_qty',
    'log_bid_qty',
    'log_ask_qty',
    'realized_spread_proxy',
    'price_impact_proxy',
    'quote_volatility_proxy',
    'flow_depth_interaction',
    'imbalance_volume_interaction',
    'depth_volume_interaction',
    'buy_sell_spread',
    'bid_ask_spread',
    'trade_informativeness',
    'execution_shortfall_proxy',
    'adverse_selection_proxy',
    'fill_probability',
    'execution_rate',
    'market_efficiency',
    'sqrt_volume',
    'sqrt_depth',
    'volume_squared',
    'imbalance_squared',
    'bid_ratio',
    'ask_ratio',
    'buy_ratio',
    'sell_ratio',
    'liquidity_consumption',
    'market_stress',
    'depth_depletion',
    'net_buying_ratio',
    'directional_volume',
    'signed_volume'
]


# feature_names = x_feaures
# feature_names = feature_names + new_features
label_name = 'label'



def get_fold_slices(n_samples):
    # 按比例划分样本索引的切片
    fold_slices = [
        {"name": "fold_1", "start": 0, "end": int(0.167 * n_samples)},
        {"name": "fold_2", "start": int(0.167 * n_samples), "end": int(0.333 * n_samples)},
        {"name": "fold_3", "start": int(0.333 * n_samples), "end": int(0.5 * n_samples)},
        {"name": "fold_4", "start": int(0.5 * n_samples), "end": int(0.667 * n_samples)},
        {"name": "fold_5", "start": int(0.667 * n_samples), "end": int(0.833 * n_samples)},
        {"name": "fold_6", "start": int(0.833 * n_samples), "end": n_samples},
    ]
    return fold_slices

# 给DataFrame添加Fold列示例
folds = get_fold_slices(len(train))
train['Fold'] = -1
for i, fs in enumerate(folds, start=1):
    train.iloc[fs["start"]:fs["end"], train.columns.get_loc("Fold")] = i



import numpy as np

def create_time_weights(n: int, decay: float = 0.95) -> np.ndarray:
    """
    Create exponentially decaying weights based on time order.
    More recent samples (later in sequence) have higher weights.
    
    Args:
        n (int): Number of samples
        decay (float): Decay factor between 0 and 1 (e.g., 0.95 means 5% decay per unit)

    Returns:
        np.ndarray: Array of weights summing to n
    """
    positions = np.arange(n)
    normalized = positions / (n - 1)
    weights = decay ** (1.0 - normalized)  # ✅ 使用 decay 参数
    return weights * n / weights.sum()



train['weight'] = create_time_weights(len(train), decay=0.95)


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

