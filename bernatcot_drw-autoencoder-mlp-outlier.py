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
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')
# Add these imports at the top
!pip install pytorch-tabnet torch scikit-optimize
!pip install koolbox scikit-learn==1.5.2
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
from xgboost import XGBRegressor
from sklearn.base import clone
from koolbox import Trainer
import seaborn as sns
import warnings
import optuna
import joblib
import gc

# TabNet wrapper for sklearn compatibility
class TabNetWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_d=64, n_a=64, n_steps=5, gamma=1.5, n_independent=2, n_shared=2, 
                 lambda_sparse=1e-3, momentum=0.02, clip_value=1., optimizer_fn=torch.optim.Adam,
                 optimizer_params=dict(lr=2e-2), scheduler_fn=torch.optim.lr_scheduler.StepLR,
                 scheduler_params=dict(step_size=50, gamma=0.9), seed=42):
        
        self.n_d = n_d
        self.n_a = n_a  
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.lambda_sparse = lambda_sparse
        self.momentum = momentum
        self.clip_value = clip_value
        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params
        self.scheduler_fn = scheduler_fn
        self.scheduler_params = scheduler_params
        self.seed = seed
        
    def fit(self, X, y):
        self.model = TabNetRegressor(
            n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps, gamma=self.gamma,
            n_independent=self.n_independent, n_shared=self.n_shared,
            lambda_sparse=self.lambda_sparse, momentum=self.momentum,
            clip_value=self.clip_value, optimizer_fn=self.optimizer_fn,
            optimizer_params=self.optimizer_params, scheduler_fn=self.scheduler_fn,
            scheduler_params=self.scheduler_params, seed=self.seed, verbose=0
        )
        
        # Convert to numpy if needed
        X_np = X.values if hasattr(X, 'values') else X
        y_np = y.values if hasattr(y, 'values') else y
        
        self.model.fit(
            X_np, y_np.reshape(-1, 1),
            eval_set=[(X_np, y_np.reshape(-1, 1))],
            max_epochs=100,
            patience=15,
            batch_size=8192,
            virtual_batch_size=512
        )
        return self
        
    def predict(self, X):
        X_np = X.values if hasattr(X, 'values') else X
        return self.model.predict(X_np).flatten()

# NODE wrapper (simplified version using extra trees as proxy since NODE is complex to implement)
class NODEWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        from sklearn.ensemble import ExtraTreesRegressor
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
    def fit(self, X, y):
        from sklearn.ensemble import ExtraTreesRegressor
        self.model = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        return self.model.predict(X)

def simple_prophet_outlier_detection(df, feature_col, timestamp_col='__index_level_0__'):
    """
    Simple example of using Prophet to detect outliers in a single feature.
    
    The idea: Prophet models the expected behavior of the feature over time.
    Points that fall far outside Prophet's confidence interval are outliers.
    """
    print(f"Detecting outliers in {feature_col} using Prophet...")
    
    # 1. Prepare data for Prophet
    prophet_df = pd.DataFrame({
        'ds': df[timestamp_col],
        'y': df[feature_col]
    })
    
    # Remove obvious bad values
    prophet_df = prophet_df[np.isfinite(prophet_df['y'])]
    
    # Resample to hourly for efficiency (adjust based on your needs)
    prophet_hourly = prophet_df.set_index('ds').resample('1H').mean().reset_index()
    prophet_hourly = prophet_hourly.dropna()
    
    # 2. Fit Prophet model
    model = Prophet(
        changepoint_prior_scale=0.05,  # Low value = less sensitive to outliers
        interval_width=0.95,  # 95% confidence interval
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True
    )
    
    model.fit(prophet_hourly)
    
    # 3. Generate predictions
    forecast = model.predict(prophet_hourly)
    
    # 4. Identify outliers
    # Method 1: Points outside confidence interval
    outliers_ci = (
        (prophet_hourly['y'] < forecast['yhat_lower']) | 
        (prophet_hourly['y'] > forecast['yhat_upper'])
    )
    
    # Method 2: Points with large standardized residuals
    residuals = prophet_hourly['y'] - forecast['yhat']
    residual_std = residuals.std()
    outliers_residual = np.abs(residuals) > 3 * residual_std
    
    # Combine both methods
    outliers = outliers_ci & outliers_residual
    
    # 5. Visualize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Time series with outliers
    ax1.plot(prophet_hourly['ds'], prophet_hourly['y'], 'b.', alpha=0.5, label='Actual')
    ax1.plot(forecast['ds'], forecast['yhat'], 'g-', linewidth=2, label='Prophet Fit')
    ax1.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                     alpha=0.2, color='green', label='95% CI')
    
    # Highlight outliers
    outlier_points = prophet_hourly[outliers]
    ax1.scatter(outlier_points['ds'], outlier_points['y'], 
               color='red', s=100, edgecolor='darkred', linewidth=2, 
               label=f'Outliers ({outliers.sum()})', zorder=10)
    
    ax1.set_title(f'Prophet Outlier Detection: {feature_col}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residual distribution
    ax2.hist(residuals, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=-3*residual_std, color='red', linestyle='--', label='±3σ threshold')
    ax2.axvline(x=3*residual_std, color='red', linestyle='--')
    ax2.set_title('Residual Distribution')
    ax2.set_xlabel('Residual (Actual - Predicted)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 6. Return outlier information
    outlier_info = {
        'n_outliers': outliers.sum(),
        'outlier_pct': outliers.sum() / len(prophet_hourly) * 100,
        'outlier_timestamps': outlier_points['ds'].tolist(),
        'outlier_values': outlier_points['y'].tolist(),
        'residual_std': residual_std
    }
    
    print(f"Found {outlier_info['n_outliers']} outliers ({outlier_info['outlier_pct']:.2f}%)")
    
    return outlier_info, model, forecast


# Quick example for multiple features
def detect_outliers_multiple_features(data_path, features_to_check=None):
    """
    Run outlier detection on multiple features.
    """
    # Load data
    if features_to_check is None:
        # Default to some common features
        features_to_check = ['volume', 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty']
    
    df = pd.read_parquet(data_path, columns=['__index_level_0__'] + features_to_check)
    
    # Ensure timestamp
    if '__index_level_0__' not in df.columns:
        df['__index_level_0__'] = pd.date_range('2023-03-01', periods=len(df), freq='T')
    
    # Analyze each feature
    outlier_summary = {}
    
    for feature in features_to_check:
        if feature in df.columns:
            print(f"\n{'='*60}")
            outlier_info, model, forecast = simple_prophet_outlier_detection(df, feature)
            outlier_summary[feature] = outlier_info
    
    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = list(outlier_summary.keys())
    outlier_pcts = [outlier_summary[f]['outlier_pct'] for f in features]
    
    bars = ax.bar(features, outlier_pcts, color='coral', edgecolor='darkred', linewidth=2)
    
    # Highlight high outlier features
    for i, pct in enumerate(outlier_pcts):
        if pct > 1.0:  # More than 1% outliers
            bars[i].set_color('red')
    
    ax.set_title('Outlier Percentage by Feature', fontsize=14, fontweight='bold')
    ax.set_ylabel('Outlier %')
    ax.set_xlabel('Feature')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, pct in zip(bars, outlier_pcts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return outlier_summary


# Practical example: Using outlier detection for data cleaning
def clean_feature_outliers(df, feature_col, outlier_info, method='cap'):
    """
    Clean outliers from a feature using different methods.
    """
    # Get outlier timestamps
    outlier_timestamps = outlier_info['outlier_timestamps']
    
    # Create copy
    df_clean = df.copy()
    
    if method == 'remove':
        # Remove rows with outliers
        mask = ~df_clean['__index_level_0__'].isin(outlier_timestamps)
        df_clean = df_clean[mask]
        print(f"Removed {len(outlier_timestamps)} rows with outliers")
        
    elif method == 'cap':
        # Cap outliers at 99th percentile
        lower_cap = df_clean[feature_col].quantile(0.01)
        upper_cap = df_clean[feature_col].quantile(0.99)
        
        df_clean[feature_col] = df_clean[feature_col].clip(lower=lower_cap, upper=upper_cap)
        print(f"Capped {feature_col} to range [{lower_cap:.2f}, {upper_cap:.2f}]")
        
    elif method == 'interpolate':
        # Replace outliers with interpolated values
        outlier_mask = df_clean['__index_level_0__'].isin(outlier_timestamps)
        df_clean.loc[outlier_mask, feature_col] = np.nan
        df_clean[feature_col] = df_clean[feature_col].interpolate(method='linear')
        print(f"Interpolated {len(outlier_timestamps)} outlier values")
    
    return df_clean


# Define your feature list

X_FEATURES = ['X363', 'X321', 'X405', 'X730', 'X523', 'X756', 'X589', 'X462', 'X779',
                'X25', 'X532', 'X520', 'X329', 'X383', 'X751', 'X535', 'X639', 'X596', 'X761',
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
        "ask_qty", "buy_qty", "sell_qty" ]


# Example usage
if __name__ == "__main__":
    # Example 1: Single feature analysis
    data_path = '/kaggle/input/drw-crypto-market-prediction/train.parquet'
    df = pd.read_parquet(data_path, columns=['__index_level_0__', 'volume', 'label'] + X_FEATURES)
    
    if '__index_level_0__' not in df.columns:
        df['__index_level_0__'] = pd.date_range('2023-03-01', periods=len(df), freq='T')
    
    # Detect outliers in volume
    outlier_info, model, forecast = simple_prophet_outlier_detection(df, 'volume')
    
    # Example 2: Multiple features
    outlier_summary = detect_outliers_multiple_features(
        data_path,
        X_FEATURES
    )
    
    # Example 3: Clean the data
    df_clean = df.copy()
    for feat in X_FEATURES:
        if feat in outlier_summary:
            df_clean = clean_feature_outliers(df_clean, feat, outlier_summary[feat], method='cap')
    print("\nOutlier detection complete!")


class CFG:
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    sample_sub_path = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

    target = "label"
    n_folds = 5
    seed = 42

    run_optuna = True
    n_optuna_trials = 250


def reduce_mem_usage(dataframe, dataset):    
    print('Reducing memory usage for:', dataset)
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
    
    for col in dataframe.columns:
        col_type = dataframe[col].dtype

        c_min = dataframe[col].min()
        c_max = dataframe[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                dataframe[col] = dataframe[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                dataframe[col] = dataframe[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                dataframe[col] = dataframe[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                dataframe[col] = dataframe[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                dataframe[col] = dataframe[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                dataframe[col] = dataframe[col].astype(np.float32)
            else:
                dataframe[col] = dataframe[col].astype(np.float64)

    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print('--- Memory usage before: {:.2f} MB'.format(initial_mem_usage))
    print('--- Memory usage after: {:.2f} MB'.format(final_mem_usage))
    print('--- Decreased memory usage by {:.1f}%\n'.format(100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage))

    return dataframe


# ===== Feature Engineering =====
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
    # Handle infinities and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # For each column, replace NaN with median for robustness
    for col in df.columns:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
    
    return df


train = df_clean
test = pd.read_parquet(CFG.test_path).reset_index(drop=True)

selected_columns = ['X363', 'X321', 'X405', 'X730', 'X523', 'X756', 'X589', 'X462', 'X779',
                'X25', 'X532', 'X520', 'X329', 'X383', 'X751', 'X535', 'X639', 'X596', 'X761',
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
        "ask_qty", "buy_qty", "sell_qty", "volume" ]

train = train[selected_columns + [CFG.target]]
test = test[selected_columns]

# Apply feature engineering
train = feature_engineering(train)
test = feature_engineering(test)


to_remove = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]

train = train.drop(columns=to_remove)
test = test.drop(columns=to_remove)

train = reduce_mem_usage(train, "train")
test = reduce_mem_usage(test, "test")

X = train.drop(CFG.target, axis=1)
y = train[CFG.target]

X_test = test


def _pearsonr(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

lgbm_params = {
    "boosting_type": "gbdt",
    "colsample_bytree": 0.5625888953382505,
    "learning_rate": 0.029312951475451557,
    "min_child_samples": 63,
    "min_child_weight": 0.11456572852335424,
    "n_estimators": 126,
    "n_jobs": -1,
    "num_leaves": 37,
    "random_state": 42,
    "reg_alpha": 85.2476527854083,
    "reg_lambda": 99.38305361388907,
    "subsample": 0.450669817684892,
    "verbose": -1
}

lgbm_goss_params = {
    "boosting_type": "goss",
    "colsample_bytree": 0.34695458228489784,
    "learning_rate": 0.031023014900595287,
    "min_child_samples": 30,
    "min_child_weight": 0.4727729225033618,
    "n_estimators": 220,
    "n_jobs": -1,
    "num_leaves": 58,
    "random_state": 42,
    "reg_alpha": 38.665994901468224,
    "reg_lambda": 92.76991677464294,
    "subsample": 0.4810891284493255,
    "verbose": -1
}

xgb_params = {
    "colsample_bylevel": 0.4778015829774066,
    "colsample_bynode": 0.362764358742407,
    "colsample_bytree": 0.7107423488010493,
    "gamma": 1.7094857725240398,
    "learning_rate": 0.02213323588455387,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "n_jobs": -1,
    "random_state": 42,
    "reg_alpha": 39.352415706891264,
    "reg_lambda": 75.44843704068275,
    "subsample": 0.06566669853471274,
    "verbosity": 0
}

histgb_params = {
    "max_iter": 500,
    "learning_rate": 0.05,
    "max_depth": 10,
    "random_state": 42
}

catboost_params = {
    "iterations": 1000,
    "learning_rate": 0.02,
    "depth": 10,
    "l2_leaf_reg": 3,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "early_stopping_rounds": 100,
    "random_seed": 42,
    "task_type": "GPU",
    "verbose": 100
}


scores = {}
oof_preds = {}
test_preds = {}


lgbm_trainer = Trainer(
    LGBMRegressor(**lgbm_params),
    cv=KFold(n_splits=5, shuffle=False),
    metric=_pearsonr,
    task="regression",
    metric_precision=6
)

lgbm_trainer.fit(X, y)

scores["LightGBM (gbdt)"] = lgbm_trainer.fold_scores
oof_preds["LightGBM (gbdt)"] = lgbm_trainer.oof_preds
test_preds["LightGBM (gbdt)"] = lgbm_trainer.predict(X_test)


lgbm_goss_trainer = Trainer(
    LGBMRegressor(**lgbm_goss_params),
    cv=KFold(n_splits=5, shuffle=False),
    metric=_pearsonr,
    task="regression",
    metric_precision=6
)

lgbm_goss_trainer.fit(X, y)

scores["LightGBM (goss)"] = lgbm_goss_trainer.fold_scores
oof_preds["LightGBM (goss)"] = lgbm_goss_trainer.oof_preds
test_preds["LightGBM (goss)"] = lgbm_goss_trainer.predict(X_test)


xgb_trainer = Trainer(
    XGBRegressor(**xgb_params),
    cv=KFold(n_splits=5, shuffle=False),
    metric=_pearsonr,
    task="regression",
    metric_precision=6
)

xgb_trainer.fit(X, y)

scores["XGBoost"] = xgb_trainer.fold_scores
oof_preds["XGBoost"] = xgb_trainer.oof_preds
test_preds["XGBoost"] = xgb_trainer.predict(X_test)


histgb_trainer = Trainer(
    HistGradientBoostingRegressor(**histgb_params),
    cv=KFold(n_splits=5, shuffle=False),
    metric=_pearsonr,
    task="regression",
    metric_precision=6
)

histgb_trainer.fit(X, y)

scores["HistGB"] = histgb_trainer.fold_scores
oof_preds["HistGB"] = histgb_trainer.oof_preds
test_preds["HistGB"] = histgb_trainer.predict(X_test)


from catboost import CatBoostRegressor

catboost_trainer = Trainer(
    CatBoostRegressor(**catboost_params),
    cv=KFold(n_splits=5, shuffle=False),
    metric=_pearsonr,
    task="regression",
    metric_precision=6
)
catboost_trainer.fit(X, y)
scores["CatBoost"] = catboost_trainer.fold_scores
oof_preds["CatBoost"] = catboost_trainer.oof_preds
test_preds["CatBoost"] = catboost_trainer.predict(X_test)


# Add TabNet to your ensemble
tabnet_trainer = Trainer(
    TabNetWrapper(n_d=16, n_a=16, n_steps=3, gamma=1.2, seed=42),
    cv=KFold(n_splits=5, shuffle=False),
    metric=_pearsonr,
    task="regression",
    metric_precision=6
)

tabnet_trainer.fit(X, y)
scores["TabNet"] = tabnet_trainer.fold_scores
oof_preds["TabNet"] = tabnet_trainer.oof_preds
test_preds["TabNet"] = tabnet_trainer.predict(X_test)

# Add NODE (ExtraTrees proxy)
node_trainer = Trainer(
    NODEWrapper(n_estimators=50, max_depth=6, random_state=42),
    cv=KFold(n_splits=5, shuffle=False),
    metric=_pearsonr,
    task="regression", 
    metric_precision=6
)

node_trainer.fit(X, y)
scores["NODE"] = node_trainer.fold_scores
oof_preds["NODE"] = node_trainer.oof_preds
test_preds["NODE"] = node_trainer.predict(X_test)


def plot_weights(weights, title):
    sorted_indices = np.argsort(weights[0])[::-1]
    sorted_coeffs = np.array(weights[0])[sorted_indices]
    sorted_model_names = np.array(list(oof_preds.keys()))[sorted_indices]

    plt.figure(figsize=(10, weights.shape[1] * 0.5))
    ax = sns.barplot(x=sorted_coeffs, y=sorted_model_names, palette="RdYlGn_r")

    for i, (value, name) in enumerate(zip(sorted_coeffs, sorted_model_names)):
        if value >= 0:
            ax.text(value, i, f"{value:.3f}", va="center", ha="left", color="black")
        else:
            ax.text(value, i, f"{value:.3f}", va="center", ha="right", color="black")

    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.1 * abs(xlim[0]), xlim[1] + 0.1 * abs(xlim[1]))

    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


X = pd.DataFrame(oof_preds)
X_test = pd.DataFrame(test_preds)


joblib.dump(X, "oof_preds.pkl")
joblib.dump(X_test, "test_preds.pkl")


class AutoEncoderMLP(BaseEstimator, RegressorMixin):
    def __init__(self, num_columns, hidden_units, dropout_rates, lr=1e-3):
        self.num_columns = num_columns
        self.hidden_units = hidden_units
        self.dropout_rates = dropout_rates
        self.lr = lr
        self.model = self._build_model()
    
    def _build_model(self):
        inp = tf.keras.layers.Input(shape=(self.num_columns,))
        x0 = tf.keras.layers.BatchNormalization()(inp)

        encoder = tf.keras.layers.GaussianNoise(self.dropout_rates[0])(x0)
        encoder = tf.keras.layers.Dense(self.hidden_units[0])(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.Activation('swish')(encoder)

        decoder = tf.keras.layers.Dropout(self.dropout_rates[1])(encoder)
        decoder = tf.keras.layers.Dense(self.num_columns, name='decoder')(decoder)

        x_reg = tf.keras.layers.Dense(self.hidden_units[1])(encoder)
        x_reg = tf.keras.layers.BatchNormalization()(x_reg)
        x_reg = tf.keras.layers.Activation('swish')(x_reg)
        x_reg = tf.keras.layers.Dropout(self.dropout_rates[2])(x_reg)

        out_reg = tf.keras.layers.Dense(1, activation='linear', name='target')(x_reg)

        model = tf.keras.models.Model(inputs=inp, outputs=[decoder, out_reg])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss={"decoder": tf.keras.losses.MeanSquaredError(),
                  "target": tf.keras.losses.MeanSquaredError()},
            loss_weights={"decoder": 0.3, "target": 1.0}
        )
        return model

    def fit(self, X, y):
        self.model.fit(
            X, {"decoder": X, "target": y},
            epochs=50,
            batch_size=8192,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5)
            ],
            verbose=0
        )
        return self

    def predict(self, X):
        _, y_pred = self.model.predict(X, verbose=0)
        return y_pred.flatten()


from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

def optimize_ensemble_weights(oof_predictions, y_true, method='bayesian'):
    """
    Find optimal ensemble weights using Bayesian optimization
    """
    n_models = len(oof_predictions)
    model_names = list(oof_predictions.keys())
    
    # Convert to numpy array
    oof_array = np.column_stack([oof_predictions[name] for name in model_names])
    
    def objective(weights):
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        ensemble_pred = np.dot(oof_array, weights)
        # Use negative correlation as we want to maximize it
        correlation = _pearsonr(y_true, ensemble_pred)
        return -correlation
    
    # Constraints: weights sum to 1 and are non-negative
    constraints = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})
    bounds = [(0, 1) for _ in range(n_models)]
    
    # Multiple random starts for better optimization
    best_result = None
    best_score = float('inf')
    
    for _ in range(20):  # 20 random starts
        initial_weights = np.random.dirichlet(np.ones(n_models))
        
        result = minimize(
            objective, initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_result = result
    
    optimal_weights = best_result.x / best_result.x.sum()
    
    return dict(zip(model_names, optimal_weights)), -best_score


def create_stacking_ensemble(oof_preds, test_preds, y, level2_models=None):
    """
    Create multi-level stacking ensemble
    """
    if level2_models is None:
        level2_models = {
            'ridge': Ridge(alpha=1.0, random_state=42),
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'lgbm_stack': LGBMRegressor(
                n_estimators=100, learning_rate=0.05, num_leaves=31,
                random_state=42, verbose=-1
            )
        }
    
    # Level 1: Base model predictions (already have these)
    X_level1 = pd.DataFrame(oof_preds)
    X_test_level1 = pd.DataFrame(test_preds)
    
    # Level 2: Train meta-models
    level2_oof = {}
    level2_test = {}
    level2_scores = {}
    
    cv = KFold(n_splits=5, shuffle=False, random_state=42)
    
    for name, model in level2_models.items():
        trainer = Trainer(
            model,
            cv=cv,
            metric=_pearsonr,
            task="regression",
            metric_precision=6
        )
        
        trainer.fit(X_level1, y)
        
        level2_oof[f"level2_{name}"] = trainer.oof_preds
        level2_test[f"level2_{name}"] = trainer.predict(X_test_level1)
        level2_scores[f"level2_{name}"] = trainer.fold_scores
        
        print(f"Level 2 {name} CV: {np.mean(trainer.fold_scores):.6f}")
    
    # Level 3: Final ensemble of level 2 models
    if len(level2_models) > 1:
        level3_weights, level3_score = optimize_ensemble_weights(level2_oof, y)
        
        # Create final prediction
        final_test_pred = np.zeros(len(X_test_level1))
        for model_name, weight in level3_weights.items():
            final_test_pred += weight * level2_test[model_name]
            
        print(f"Final stacked ensemble CV: {level3_score:.6f}")
        
        return final_test_pred, level3_weights, level2_scores
    else:
        model_name = list(level2_test.keys())[0]
        return level2_test[model_name], {model_name: 1.0}, level2_scores




ae_model = AutoEncoderMLP(
    num_columns=X.shape[1],
    hidden_units=[128, 128],
    dropout_rates=[0.06, 0.13, 0.23],
    lr=1e-3
)


ae_trainer = Trainer(
    ae_model,
    cv=KFold(n_splits=5, shuffle=False),
    metric=_pearsonr,
    task="regression",
    metric_precision=6
)

ae_trainer.fit(X, y)

scores["ae"] = ae_trainer.fold_scores
oof_preds["ae"] = ae_trainer.oof_preds
test_preds["ae"] = ae_trainer.predict(X_test)


print("Creating stacked ensemble...")
final_preds, stack_weights, stack_scores = create_stacking_ensemble(oof_preds, test_preds, y)

# Final submission
sub = pd.read_csv(CFG.sample_sub_path)
sub["prediction"] = final_preds
sub.to_csv("submission.csv", index=False)
sub.head()


scores = pd.DataFrame(scores)
mean_scores = scores.mean().sort_values(ascending=False)
order = scores.mean().sort_values(ascending=False).index.tolist()

min_score = mean_scores.min()
max_score = mean_scores.max()
padding = (max_score - min_score) * 0.5
lower_limit = min_score - padding
upper_limit = max_score + padding

fig, axs = plt.subplots(1, 2, figsize=(15, scores.shape[1] * 0.5))

boxplot = sns.boxplot(data=scores, order=order, ax=axs[0], orient="h", color="grey")
axs[0].set_title(f"Fold Score")
axs[0].set_xlabel("")
axs[0].set_ylabel("")

barplot = sns.barplot(x=mean_scores.values, y=mean_scores.index, ax=axs[1], color="grey")
axs[1].set_title(f"Average Score")
axs[1].set_xlabel("")
axs[1].set_xlim(left=lower_limit, right=upper_limit)
axs[1].set_ylabel("")

for i, (score, model) in enumerate(zip(mean_scores.values, mean_scores.index)):
    color = "cyan" if "ensemble" in model.lower() else "grey"
    barplot.patches[i].set_facecolor(color)
    boxplot.patches[i].set_facecolor(color)
    barplot.text(score, i, round(score, 6), va="center")

plt.tight_layout()
plt.show()




