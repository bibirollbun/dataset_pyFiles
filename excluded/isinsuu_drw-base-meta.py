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
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Config
class CFG:
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    sample_sub_path = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    target = "label"

# 1. Load data
X_FEATURES = [
    "X752",
    "X287",
    "X298",
    "X759",
    "X302",
    "X55",
    "X56",
    "X52",
    "X303",
    "X51",
    "X598", "X385", "X603", "X674",
    "X415", "X345", "X174", "X178", "X168", "X612", "bid_qty",
        "ask_qty", "buy_qty", "sell_qty", "volume"
    ]



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
    # Handle infinities and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # For each column, replace NaN with median for robustness
    for col in df.columns:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
    
    return df


train = pd.read_parquet(CFG.train_path, columns=['__index_level_0__', CFG.target] + X_FEATURES)
X_test = pd.read_parquet(CFG.test_path, columns=X_FEATURES)


X_train = train[X_FEATURES]
y_train = train[CFG.target]

X_train = feature_engineering(X_train)
X_test = feature_engineering(X_test)

from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from scipy.stats import pearsonr

base_models = [
    ("xgb", XGBRegressor(
        tree_method="hist",
        device="gpu",
        colsample_bylevel=0.4778,
        colsample_bynode=0.3628,
        colsample_bytree=0.7107,
        gamma=1.7095,
        learning_rate=0.02213,
        max_depth=20,
        max_leaves=12,
        min_child_weight=16,
        n_estimators=1667,
        subsample=0.06567,
        reg_alpha=39.3524,
        reg_lambda=75.4484,
        verbosity=0,
        random_state=42,
        n_jobs=-1
    )),
    
    ("lgbm", LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=10,
        reg_lambda=10,
        random_state=42,
        device="gpu",
        verbosity=-1,
        n_jobs=-1
    )),
    
    ("ridge", Ridge(alpha=1.0))
]

meta_model = Ridge(alpha=1.0)

stack_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=KFold(n_splits=5, shuffle=False),
    passthrough=True,
    n_jobs=-1
)

stack_model.fit(X_train, y_train)

y_pred = stack_model.predict(X_train)
print("Stacked R²:", r2_score(y_train, y_pred))
print("Stacked MSE:", mean_squared_error(y_train, y_pred))
pearson_corr, _ = pearsonr(y_train, y_pred)
print("Pearson Correlation (train):", round(pearson_corr, 6))

test_pred = stack_model.predict(X_test)

sub = pd.read_csv(CFG.sample_sub_path)
sub["prediction"] = test_pred
sub.to_csv("submission.csv", index=False)
print("Submission saved as: submission.csv")

