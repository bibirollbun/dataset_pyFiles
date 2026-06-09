import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin


import optuna

import xgboost as xgb
import time
from scipy.stats import pearsonr

import gc 
import functools

from pathlib import Path
from tqdm.auto import tqdm
from typing import Union, List


def _pearsonr(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]
    
def pearsonr_coeff(preds, data):
    y_true = data.get_label()
    valid_score = _pearsonr(y_true, preds)
    return 'pearsonr_coeff_score', valid_score



class VotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators

    def fit(self, X, y=None):
        # No training in VotingModel since estimators are pre-trained
        return self

    def predict(self, X):
        y_preds = []
        for estimator in self.estimators:
            # If the estimator is an XGBoost Booster, convert DataFrame to DMatrix
            if isinstance(estimator, xgb.core.Booster):
                X_converted = xgb.DMatrix(X)
            else:
                X_converted = X
            y_preds.append(estimator.predict(X_converted))
        return np.mean(y_preds, axis=0)

    def predict_proba(self, X):
        y_preds = []
        for estimator in self.estimators:
            if hasattr(estimator, "predict_proba"):
                y_preds.append(estimator.predict_proba(X))
            else:
                raise AttributeError("One of the estimators does not support predict_proba.")
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


class Config:
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    FEATURES = ['X363', 'X321', 'X405', 'X730', 'X523', 'X756', 'X589', 'X462', 'X779',
                'X25', 'X532', 'X520', 'X329', 'X383', 'X751', 'X535', 'X639', 'X596', 'X761',
                'X752', 'X287', 'X298', 'X759', 'X302', 'X55', 'X56', 'X52', 'X303', 'X51',
                'X598', 'X385', 'X603', 'X674', 'X415', 'X345', 'X174', 'X178', 'X168', 'X612',
                'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']

    LABEL_COLUMN = "label"
    RANDOM_STATE = 42


def feature_engineering(df):
    # Create a stable working copy to avoid modifying the original DataFrame
    df = df.copy()
    epsilon = np.finfo(float).eps  # Very small constant to prevent divide-by-zero errors

    # Replace NaNs with zeros to prevent issues during feature construction
    df.fillna(0, inplace=True)

    # Interaction Features
    # Capture multiplicative effects between different order quantities
    new_features = {
        'bid_ask_interaction': df['bid_qty'] * df['ask_qty'],
        'bid_buy_interaction': df['bid_qty'] * df['buy_qty'],
        'bid_sell_interaction': df['bid_qty'] * df['sell_qty'],
        'ask_buy_interaction': df['ask_qty'] * df['buy_qty'],
        'ask_sell_interaction': df['ask_qty'] * df['sell_qty'],
        'buy_sell_interaction': df['buy_qty'] * df['sell_qty'],

        # Spread Indicator
        # Measures the difference between ask and bid sizes normalized by their total
        'spread_indicator': (df['ask_qty'] - df['bid_qty']) / (df['ask_qty'] + df['bid_qty'] + epsilon),

        # Volume-Weighted Quantities
        'volume_weighted_buy': df['buy_qty'] * df['volume'],
        'volume_weighted_sell': df['sell_qty'] * df['volume'],
        'volume_weighted_bid': df['bid_qty'] * df['volume'],
        'volume_weighted_ask': df['ask_qty'] * df['volume'],

        # Ratio Features
        'buy_sell_ratio': df['buy_qty'] / (df['sell_qty'] + epsilon),
        'bid_ask_ratio': df['bid_qty'] / (df['ask_qty'] + epsilon),

        # Order Flow & Market Pressure
        'order_flow_imbalance': (df['buy_qty'] - df['sell_qty']) / (df['volume'] + epsilon),
        'buying_pressure': df['buy_qty'] / (df['volume'] + epsilon),
        'selling_pressure': df['sell_qty'] / (df['volume'] + epsilon)
    }

    # Add new features into the DataFrame efficiently
    df = df.assign(**new_features)

    # Liquidity Measures
    df['total_liquidity'] = df['bid_qty'] + df['ask_qty']
    df['liquidity_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_liquidity'] + epsilon)
    df['relative_spread'] = (df['ask_qty'] - df['bid_qty']) / (df['volume'] + epsilon)

    # Trade Activity & Intensity
    df['trade_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + epsilon)
    df['avg_trade_size'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + epsilon)
    df['net_trade_flow'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + epsilon)

    # Market Depth & Engagement
    df['depth_ratio'] = df['total_liquidity'] / (df['volume'] + epsilon)
    df['volume_participation'] = (df['buy_qty'] + df['sell_qty']) / (df['total_liquidity'] + epsilon)
    df['market_activity'] = df['volume'] * df['total_liquidity']

    # Execution Quality Metrics
    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + epsilon)
    df['realized_volatility_proxy'] = np.abs(df['order_flow_imbalance']) * df['volume']

    # Normalized Volumes
    df['normalized_buy_volume'] = df['buy_qty'] / (df['bid_qty'] + epsilon)
    df['normalized_sell_volume'] = df['sell_qty'] / (df['ask_qty'] + epsilon)

    # Advanced Interactions
    df['liquidity_adjusted_imbalance'] = df['order_flow_imbalance'] * df['depth_ratio']
    df['pressure_spread_interaction'] = df['buying_pressure'] * df['spread_indicator']

    # Depth Ratio & Imbalance
    df['bid_depth_ratio'] = df['bid_qty'] / (df['volume'] + 1e-8)
    df['ask_depth_ratio'] = df['ask_qty'] / (df['volume'] + 1e-8)
    df['depth_imbalance'] = (df['bid_depth_ratio'] - df['ask_depth_ratio']) / (df['depth_ratio'] + 1e-8)

    # Illiquidity Metrics
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + 1e-8)
    df['net_pressure'] = df['buying_pressure'] - df['selling_pressure']
    df['amihud_illiquidity'] = np.abs(df['net_pressure']) / (df['volume'] + 1e-8)
    df['liquidity_consumption'] = df['volume'] / (df['total_liquidity'] + 1e-8)

    # Price & Execution Efficiency
    df['bid_ask_spread'] = df['ask_qty'] - df['bid_qty']
    df['price_efficiency'] = 1 / (1 + df['amihud_illiquidity'])
    df['execution_quality'] = df['volume'] / (df['bid_ask_spread'] + 1)

    # Toxicity Proxies
    df['pin_proxy'] = np.abs(df['order_flow_imbalance']) * df['amihud_illiquidity']
    df['order_toxicity'] = np.abs(df['order_flow_imbalance']) * df['kyle_lambda']

    # Market Momentum Features
    df['bid_momentum'] = df['bid_qty'] * df['buy_qty'] / (df['volume'] + 1e-8)
    df['ask_momentum'] = df['ask_qty'] * df['sell_qty'] / (df['volume'] + 1e-8)
    df['liquidity_adjusted_volume'] = df['volume'] / np.sqrt(df['total_liquidity'] + 1)

    # Log-Transformed Stability Features
    df['log_volume'] = np.log1p(df['volume'])
    df['log_liquidity'] = np.log1p(df['total_liquidity'])
    df['log_spread'] = np.log1p(np.abs(df['bid_ask_spread']))

    # Final Sanity Cleanup
    # Replace infinities with NaNs, then backfill with zero
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df



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


train_df = reduce_mem_usage(pd.read_parquet(Config.TRAIN_PATH))
test_df = reduce_mem_usage(pd.read_parquet(Config.TEST_PATH))
submission_df = pd.read_csv(Config.SUBMISSION_PATH)

# Apply feature engineering
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# Assign folds
train_df.loc['2023-03-01 00:00:00':'2023-05-01 00:00:00', 'Fold'] = 1
train_df.loc['2023-05-01 00:00:00':'2023-07-01 00:00:00', 'Fold'] = 2
train_df.loc['2023-07-01 00:00:00':'2023-09-01 00:00:00', 'Fold'] = 3
train_df.loc['2023-09-01 00:00:00':'2023-11-01 00:00:00', 'Fold'] = 4
train_df.loc['2023-11-01 00:00:00':'2024-01-01 00:00:00', 'Fold'] = 5
train_df.loc['2024-01-01 00:00:00':'2024-03-01 00:00:00', 'Fold'] = 6

# Apply time weights
train_df['weight'] = create_time_weights(len(train_df))


models = []
valid_scores = []

# Objective function for hyperparameter tuning
def objective(trial):
    fold_scores = []
    
    for fold in range(1, 6):
        # Split data into train/validation based on fold
        X_train = train_df[train_df["Fold"] != fold][Config.FEATURES]
        w_train = train_df[train_df["Fold"] != fold]["weight"]
        y_train = train_df[train_df["Fold"] != fold][Config.LABEL_COLUMN]

        X_valid = train_df[train_df["Fold"] == fold][Config.FEATURES]
        w_valid = train_df[train_df["Fold"] == fold]["weight"]
        y_valid = train_df[train_df["Fold"] == fold][Config.LABEL_COLUMN]

        dtrain = xgb.DMatrix(data=X_train, label=y_train, weight=w_train)
        dvalid = xgb.DMatrix(data=X_valid, label=y_valid, weight=w_valid)

        # Suggest hyperparameters
        params = {
            "objective": "reg:squarederror",
            "seed": Config.RANDOM_STATE,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1),
            "gamma": trial.suggest_float("gamma", 0, 10),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        }

        num_boost_round = 150
        evals = [(dtrain, "train"), (dvalid, "valid")]

        model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=evals,custom_metric=pearsonr_coeff, verbose_eval=False)
        valid_pred = model.predict(dvalid)

        # Pearson correlation as performance metric
        fold_score = _pearsonr(y_valid, valid_pred)
        fold_scores.append(fold_score)
        
        models.append(model)

    # Return average Pearson correlation across folds
    return np.mean(fold_scores)

# Run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)

# Print best parameters found
print("Best parameters:", study.best_params)



# Initialize the VotingModel with trained estimators; notice the variable is 'voting_model'
voting_model = VotingModel(models)

# Option 1: If you want to work manually with DMatrix, you can do:
# dtest = xgb.DMatrix(test_df[Config.FEATURES])
# However, if using VotingModel, it's better to let it handle conversion:
# submission['prediction'] = voting_model.predict(dtest)  # if predict() expects DMatrix
# Option 2: Use DataFrame directly, and let VotingModel automatically convert if needed:
submission_df['prediction'] = voting_model.predict(test_df[Config.FEATURES])

# Save predictions to CSV file
submission_df.to_csv(r'submission.csv', index=False)


