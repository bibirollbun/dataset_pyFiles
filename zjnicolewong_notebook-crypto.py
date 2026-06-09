# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


class Config:
    # Data paths (Kaggle environment)
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

    # Selected features
    FEATURES = [
        "X863", "X345", "X612", "X855", "X174",
        "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"
    ]

    # Lag windows
    LAGS = [1, 5, 10]

    # Model parameters
    ALPHA = 0.1
    N_FOLDS = 3


def create_time_weights(n):
    """Linearly increasing weights for recent samples"""
    weights = np.linspace(0.1, 1.0, n)  # make sure != 0
    return weights / weights.sum()


def add_features(df):
    """
    Add simple derived features and lagged versions of core columns
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    # Feature interactions
    df['liquidity_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-8)
    df['order_flow'] = (df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)

    # Lagged features
    for col in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']:
        for lag in Config.LAGS:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # Fill missing values efficiently
    return df.ffill().bfill()


def load_data():
    """
    Load and preprocess training and test data
    
    Returns:
        X_train, y_train, X_test arrays
    """
    train_df = pd.read_parquet(Config.TRAIN_PATH, columns=Config.FEATURES + ["label"])
    test_df = pd.read_parquet(Config.TEST_PATH, columns=Config.FEATURES)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    # Feature engineering
    train_df = add_features(train_df)
    test_df = add_features(test_df)

    feature_cols = [col for col in train_df.columns if col != "label"]

    # Clean inf values
    train_df[feature_cols] = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    test_df[feature_cols] = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    y_train = train_df["label"].values

    return X_train, y_train, X_test


def train_model(X, y, X_test):
    tscv = TimeSeriesSplit(n_splits=Config.N_FOLDS)
    weights = create_time_weights(len(X))
    preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"Training Fold {fold+1}/{Config.N_FOLDS}")

        fold_weights = weights[train_idx]

        model = Ridge(alpha=Config.ALPHA)
        if fold_weights.sum() > 0:
            model.fit(X[train_idx], y[train_idx], sample_weight=fold_weights)
        else:
            print("Skipping weights for this fold.")
            model.fit(X[train_idx], y[train_idx])

        preds += model.predict(X_test)

    return preds / Config.N_FOLDS


def make_submission(preds):
    """
    Generate submission file
    
    Args:
        preds: Predicted values
    """
    submission = pd.read_csv(Config.SUBMISSION_PATH)
    submission["prediction"] = preds
    submission.to_csv("submission.csv", index=False)
    print("Submission file created")


if __name__ == "__main__":
    X_train, y_train, X_test = load_data()
    predictions = train_model(X_train, y_train, X_test)
    make_submission(predictions)

