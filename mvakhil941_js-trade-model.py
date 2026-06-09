import numpy as np
import pandas as pd
import os

input_dir = "/kaggle/input"
for root, _, files in os.walk(input_dir):
    for file in files:
        print(os.path.join(root, file))


import pickle
import lightgbm as lgb
import kaggle_evaluation.jane_street_inference_server
import warnings
warnings.filterwarnings('ignore')


# CONFIGURATION

THRESHOLD = 0.0  # Trade if predicted return > 0
FEATURE_COLS = [f'feature_{i:02d}' for i in range(79)]


# LOAD MODEL

print("Loading LightGBM model")

try:
    model = lgb.Booster(model_file='/kaggle/input/jane-street-trading-model-v1/models/lgb_model.txt')
    print("LightGBM model loaded successfully")
except Exception as e:
    print(f"ERROR: Could not load model: {e}")
    raise


# FEATURE ENGINEERING FOR INFERENCE

class InferenceFeatureEngineer:
    """
    Lightweight feature engineering for real-time inference.
    Recreates the same features used during training.
    """

    def __init__(self):
        self.feature_cols = FEATURE_COLS

    def transform(self, df):
        """Apply feature engineering optimized for speed"""
        df_processed = df.copy()

        # Fill missing values with 0
        for col in self.feature_cols:
            if col in df_processed.columns:
                if df_processed[col].isnull().any():
                    df_processed[col].fillna(0, inplace=True)

        # Basic statistical features
        available_features = [col for col in self.feature_cols if col in df_processed.columns]
        if len(available_features) > 0:
            feature_data = df_processed[available_features]
            df_processed['feature_mean'] = feature_data.mean(axis=1)
            df_processed['feature_std'] = feature_data.std(axis=1).fillna(0)
            df_processed['feature_max'] = feature_data.max(axis=1)
            df_processed['feature_min'] = feature_data.min(axis=1)
            df_processed['feature_range'] = df_processed['feature_max'] - df_processed['feature_min']
            df_processed['feature_count'] = feature_data.notna().sum(axis=1)

        # CRITICAL: Advanced interaction features

        # Time interactions
        if 'feature_06' in df_processed.columns and 'time_id' in df_processed.columns:
            df_processed['f06_x_time'] = df_processed['feature_06'] * df_processed['time_id']

        if 'feature_07' in df_processed.columns and 'time_id' in df_processed.columns:
            df_processed['f07_x_time'] = df_processed['feature_07'] * df_processed['time_id']

        # Polynomial features
        if 'feature_06' in df_processed.columns:
            df_processed['f06_squared'] = df_processed['feature_06'] ** 2
            df_processed['f06_cubed'] = df_processed['feature_06'] ** 3

        if 'feature_07' in df_processed.columns:
            df_processed['f07_squared'] = df_processed['feature_07'] ** 2

        # Cross-feature interactions
        if 'feature_06' in df_processed.columns and 'feature_07' in df_processed.columns:
            df_processed['f06_x_f07'] = df_processed['feature_06'] * df_processed['feature_07']
            df_processed['f06_div_f07'] = df_processed['feature_06'] / (df_processed['feature_07'].abs() + 1e-5)

        if 'feature_06' in df_processed.columns and 'feature_05' in df_processed.columns:
            df_processed['f06_x_f05'] = df_processed['feature_06'] * df_processed['feature_05']

        if 'feature_07' in df_processed.columns and 'feature_05' in df_processed.columns:
            df_processed['f07_x_f05'] = df_processed['feature_07'] * df_processed['feature_05']

        if 'feature_05' in df_processed.columns and 'feature_07' in df_processed.columns:
            df_processed['f05_div_f07'] = df_processed['feature_05'] / (df_processed['feature_07'].abs() + 1e-5)

        # Deviation features
        if 'feature_06' in df_processed.columns and 'feature_mean' in df_processed.columns:
            df_processed['f06_vs_mean'] = df_processed['feature_06'] / (df_processed['feature_mean'].abs() + 1e-5)

        if 'feature_06' in df_processed.columns and 'feature_std' in df_processed.columns:
            df_processed['f06_vs_std'] = df_processed['feature_06'] / (df_processed['feature_std'] + 1e-5)

        # 4. Time features (cyclical encoding - captures daily patterns)
        if 'time_id' in df_processed.columns:
            time_max = 1000  # Approximate max time_id
            df_processed['time_normalized'] = df_processed['time_id'] / time_max
            df_processed['time_sin'] = np.sin(2 * np.pi * df_processed['time_normalized'])
            df_processed['time_cos'] = np.cos(2 * np.pi * df_processed['time_normalized'])
            df_processed['time_period'] = (df_processed['time_normalized'] * 3).astype(int).clip(0, 2).astype(float)

        return df_processed

    def prepare_features(self, df):
        """Prepare features for model prediction"""
        
        feature_columns = [col for col in df.columns if (
            col.startswith('feature_') or
            col.startswith('f0') or  # Engineered features like f06_x_time
            col.startswith('time_') or
            col in ['feature_mean', 'feature_std', 'feature_max', 'feature_min',
                   'feature_range', 'feature_count', 'symbol_id', 'time_id', 'weight']
        )]

        # Remove duplicates while preserving order
        feature_columns = list(dict.fromkeys(feature_columns))

        # Extract features
        X = df[feature_columns].copy()

        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)

        # Final missing value check
        X = X.fillna(0)

        return X

# Initialize feature engineer
engineer = InferenceFeatureEngineer()
print("Feature engineer initialized")


# PREDICTION FUNCTION 

# Global counters for tracking
prediction_count = 0
trade_count = 0

def predict(test_df, lags_df):
    """
    Main prediction function called by Jane Street inference server.

    Args:
        test_df: DataFrame with current test features
        lags_df: DataFrame with lagged features (optional, not used here)

    Returns:
        DataFrame with prediction (responder_6 column)
    """
    global prediction_count, trade_count

    try:
        # feature engineering
        df_transformed = engineer.transform(test_df)

        # feature matrix
        X = engineer.prepare_features(df_transformed)

        # prediction using LightGBM model
        prediction = model.predict(X.values)[0]

        # trading decision
        decision = 1 if prediction > THRESHOLD else 0

        # track statistics
        prediction_count += 1
        if decision == 1:
            trade_count += 1

        # progress update (every 1000 predictions)
        if prediction_count % 1000 == 0:
            trade_rate = trade_count / prediction_count * 100
            print(f"Predictions: {prediction_count:,} | Trades: {trade_count:,} ({trade_rate:.1f}%)")

        # return prediction in required format
        return pd.DataFrame({'responder_6': [decision]})

    except Exception as e:
        print(f" Prediction error: {e}")
        import traceback
        traceback.print_exc()
        # Return safe default (no trade on error)
        return pd.DataFrame({'responder_6': [0]})


# INITIALIZE AND RUN INFERENCE SERVER

print("="*80)
print("Starting Jane Street inference server")
print("="*80)

# Initialize the Jane Street inference server with our predict function
inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

# Simply call serve() - it handles both test and competition modes automatically
inference_server.serve()


# SUMMARY

print("\n" + "="*80)
print("SUBMISSION COMPLETE")
print("="*80)
print(f"Total predictions: {prediction_count:,}")
if prediction_count > 0:
    trade_rate = trade_count / prediction_count * 100
    print(f"Trades executed: {trade_count:,} ({trade_rate:.1f}%)")
    print(f"\n Performance summary:")
    print(f"   - Trade rate: {trade_rate:.1f}% (target: 45-50%)")
    if 40 <= trade_rate <= 60:
        print(f"   - Status:  GOOD - Conservative strategy")
    elif trade_rate > 60:
        print(f"   - Status:   WARNING - Too aggressive, consider increasing threshold")
    else:
        print(f"   - Status:   WARNING - Too conservative, consider decreasing threshold")
print("="*80)

