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


import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
# from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr


def min_max_rolling(df,col,rolling=30):
    roll_min = df[col].rolling(rolling).min()
    roll_max = df[col].rolling(rolling).max()
    current = df[col]

    return (current-roll_min)/(roll_max-roll_min)

def standardize_rolling(df,col,rolling=30):
    roll_mean = df[col].rolling(rolling).mean()
    roll_std = df[col].rolling(rolling).std()
    current = df[col]

    return (current-roll_mean)/(roll_std)

def feature_engineering(df):
    # Stationary Features from Previous Version
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-10)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-10)
    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-10)
    df['normalized_net_flow'] = (df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-10)
    df['total_depth'] = df['bid_qty'] + df['ask_qty']  # Temporary for calculations
    df['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['relative_spread'] = np.abs(df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['kyle_lambda'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
    df['flow_toxicity'] = np.abs(df['order_flow_imbalance']) * df['volume']
    df['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['volume_depth_ratio'] = df['volume'] / (df['total_depth'] + 1e-10)
    df['activity_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-10)
    df['realized_spread_proxy'] = 2 * np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
    df['price_impact_proxy'] = (df['buy_qty'] - df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['quote_volatility_proxy'] = np.abs(df['depth_imbalance'])
    df['imbalance_volume_interaction'] = df['order_flow_imbalance'] * df['volume']
    df['trade_informativeness'] = (df['buy_qty'] - df['sell_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['execution_shortfall_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
    df['adverse_selection_proxy'] = (df['buy_qty'] - df['sell_qty']) / (df['total_depth'] + 1e-10) * df['volume']
    df['fill_probability'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['execution_rate'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_efficiency'] = df['volume'] / (np.abs(df['bid_qty'] - df['ask_qty']) + 1e-10)
    df['imbalance_squared'] = df['order_flow_imbalance'] ** 2
    df['bid_ratio'] = df['bid_qty'] / (df['total_depth'] + 1e-10)
    df['ask_ratio'] = df['ask_qty'] / (df['total_depth'] + 1e-10)
    df['buy_ratio'] = df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['sell_ratio'] = df['sell_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['liquidity_consumption'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_stress'] = df['volume_depth_ratio'] * np.abs(df['order_flow_imbalance'])
    df['depth_depletion'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['net_buying_ratio'] = (df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)

    # Previous New Features
    df['order_flow_momentum'] = df['order_flow_imbalance'].diff() / (df['volume'] + 1e-10)
    df['volume_rolling_mean'] = df['volume'].rolling(window=5).mean()
    df['relative_volume_change'] = (df['volume'] - df['volume_rolling_mean']) / (df['volume_rolling_mean'] + 1e-10)
    df['spread_volatility'] = df['effective_spread_proxy'].rolling(window=5).std() / (df['effective_spread_proxy'].rolling(window=5).mean() + 1e-10)
    df['depth_turnover'] = df['volume'] / (df['total_depth'] + 1e-10)
    df['imbalance_persistence'] = df['order_flow_imbalance'].shift(1) / (df['order_flow_imbalance'] + 1e-10)
    df['volume_diff'] = df['volume'].diff()
    df['volume_acceleration'] = df['volume_diff'].diff() / (df['volume_rolling_mean'] + 1e-10)
    df['normalized_price_impact'] = (df['buy_qty'] - df['sell_qty']) / (df['total_depth'] + df['volume'] + 1e-10)
    df['liquidity_asymmetry'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)

    # New Noise-Reducing Features
    # 1. Smoothed Order Flow Imbalance (EMA)
    df['smoothed_order_flow'] = df['order_flow_imbalance'].ewm(span=30, adjust=False).mean()
    
    # 2. Volatility-Adjusted Order Flow
    df['order_flow_volatility'] = df['order_flow_imbalance'].rolling(window=30).std()
    df['volatility_adjusted_flow'] = df['order_flow_imbalance'] / (df['order_flow_volatility'] + 1e-10)
    
    # 3. Rank-Based Volume (robust to outliers)
    df['volume_rank'] = min_max_rolling(df,'volume',rolling=30)
    
    # 4. Lagged Bid-Ask Imbalance
    df['lagged_bid_ask_imbalance'] = df['bid_ask_imbalance'].shift(1)
    df['lagged_imbalance_change'] = (df['bid_ask_imbalance'] - df['lagged_bid_ask_imbalance']) / (df['total_depth'] + 1e-10)
    
    # 5. Volatility-Adjusted Spread
    df['spread_volatility_rolling'] = df['effective_spread_proxy'].rolling(window=5).std()
    df['volatility_adjusted_spread'] = df['effective_spread_proxy'] / (df['spread_volatility_rolling'] + 1e-10)
    
    # 6. Smoothed Depth Turnover (EMA)
    df['smoothed_depth_turnover'] = df['depth_turnover'].ewm(span=30, adjust=False).mean()
    
    # 7. Relative Momentum of Net Buying Ratio
    df['net_buying_momentum'] = df['net_buying_ratio'].diff() / (df['volume_rolling_mean'] + 1e-10)
    
    # 8. Robust Imbalance Ratio (using quantiles)
    df['imbalance_quantile'] = df['order_flow_imbalance'].rolling(window=30).median()
    df['robust_imbalance_ratio'] = df['order_flow_imbalance'] / (df['imbalance_quantile'] + 1e-10)

    # Replace infinities and NaNs
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Drop temporary columns
    df = df.drop(columns=['total_depth', 'volume_rolling_mean', 'volume_diff', 
                         'order_flow_volatility', 'spread_volatility_rolling', 
                         'lagged_bid_ask_imbalance','imbalance_quantile'], errors='ignore')
    
    return df



import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from scipy.stats import pearsonr
import pickle
import os

# Configuration
class Config:
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    FEATURES = ['X758', "X752", "X287", "X298", "X759", "X302", "X55", "X56", "X52", "X303", "X51", "X598", "X385", "X603", "X674", "X415", "X345", "X174",
                "X178", "X168", "X612", "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume" ]
    SELECTED_FEATURES = [
        'X758',    
        # Base features (excluding non-stationary volume, bid_qty, ask_qty, buy_qty, sell_qty)
        "X752", "X287", "X298", "X759", "X302", "X55", "X56", "X52", "X303", "X51",
        "X598", "X385", "X603", "X674", "X415", "X345", "X174", "X178", "X168", "X612", 'volume_rank',
        # Engineered stationary features
        "buy_sell_ratio", "selling_pressure", "effective_spread_proxy", "bid_ask_imbalance",
        "order_flow_imbalance", "liquidity_ratio", "normalized_net_flow", "buying_pressure",
        "depth_imbalance", "relative_spread", "kyle_lambda", "flow_toxicity",
        "aggressive_flow_ratio", "volume_depth_ratio", "activity_intensity",
        "realized_spread_proxy", "price_impact_proxy", "quote_volatility_proxy",
        "imbalance_volume_interaction", "trade_informativeness", "execution_shortfall_proxy",
        "adverse_selection_proxy", "fill_probability", "execution_rate", "market_efficiency",
        "imbalance_squared", "bid_ratio", "ask_ratio", "buy_ratio", "sell_ratio",
        "liquidity_consumption", "market_stress", "depth_depletion", "net_buying_ratio",
        # Previous new features
        "order_flow_momentum", "relative_volume_change", "spread_volatility",
        "depth_turnover", "imbalance_persistence", "volume_acceleration",
        "normalized_price_impact", "liquidity_asymmetry",
        # New noise-reducing features
        "smoothed_order_flow", "volatility_adjusted_flow", 
        "lagged_imbalance_change", "volatility_adjusted_spread", "smoothed_depth_turnover",
        "net_buying_momentum", "robust_imbalance_ratio"
    ]
    
    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42
    EMBARGO_SIZE = 100
    CATBOOST_PARAMS = {
"learning_rate": 0.03393830807147021,
"depth": 4,
"l2_leaf_reg": 7.158097238609412,
"iterations": 936,
"subsample": 0.5610191174223894,
"min_data_in_leaf": 54,
"task_type": "GPU",
"bootstrap_type": "Bernoulli",
"random_seed": 42,
"verbose": 0
}

def load_data():
    train_df = pd.read_parquet(Config.TRAIN_PATH, columns=Config.FEATURES + [Config.LABEL_COLUMN])
    test_df = pd.read_parquet(Config.TEST_PATH, columns=Config.FEATURES)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    print(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}, Submission: {submission_df.shape}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df

def create_time_decay_weights(n: int, decay: float = 0.9) -> np.ndarray:
    positions = np.arange(n)
    normalized = positions / (n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()

def train_and_evaluate(train_df, test_df):
    n_samples = len(train_df)
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)
    oof_preds = np.zeros(n_samples)
    test_preds = np.zeros(len(test_df))
    trained_models = []

    os.makedirs("models", exist_ok=True)
    full_weights = create_time_decay_weights(n_samples)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), 1):
        print(f"Training Fold {fold}/{Config.N_FOLDS}")

        # Apply embargo
        embargo_end = valid_idx[-1] + Config.EMBARGO_SIZE + 1 if valid_idx[-1] < n_samples - 1 else n_samples
        embargo_mask = (train_idx < valid_idx[0]) | (train_idx >= embargo_end)
        train_idx_embargoed = train_idx[embargo_mask]

        # Prepare data
        X_train = train_df.iloc[train_idx_embargoed][Config.SELECTED_FEATURES]
        y_train = train_df.iloc[train_idx_embargoed][Config.LABEL_COLUMN]
        X_valid = train_df.iloc[valid_idx][Config.SELECTED_FEATURES]
        y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN]
        sample_weights = full_weights[train_idx_embargoed]

        # Train model
        model = CatBoostRegressor(**Config.CATBOOST_PARAMS)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=50
        )

        # Save model
        model_path = f"models/catboost_fold_{fold}.cbm"
        model.save_model(model_path)
        trained_models.append(model_path)

        # Out-of-fold predictions
        oof_preds[valid_idx] = model.predict(X_valid)

        # Test predictions
        test_preds += model.predict(test_df[Config.SELECTED_FEATURES]) / Config.N_FOLDS

        # Evaluate fold
        fold_score = pearsonr(y_valid, oof_preds[valid_idx])[0]
        print(f"Fold {fold} Pearson Correlation: {fold_score:.4f}")

    oof_score = pearsonr(train_df[Config.LABEL_COLUMN], oof_preds)[0]
    print(f"\nOverall OOF Pearson Correlation: {oof_score:.4f}")

    return oof_preds, test_preds, trained_models

def load_and_predict(test_df, model_paths):
    test_preds = np.zeros(len(test_df))
    for model_path in model_paths:
        model = CatBoostRegressor()
        model.load_model(model_path)
        test_preds += model.predict(test_df[Config.SELECTED_FEATURES]) / len(model_paths)
    return test_preds

def main():
    # Load data (assuming feature_engineering is defined elsewhere)
    train_df, test_df, submission_df = load_data()

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # Train and evaluate
    oof_preds, test_preds, model_paths = train_and_evaluate(train_df, test_df)

    # Generate submission
    submission_df["prediction"] = test_preds
    submission_df.to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    main()

