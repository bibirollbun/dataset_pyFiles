import numpy as np
import pandas as pd
import os

input_dir = "/kaggle/input"
for root, _, files in os.walk(input_dir):
    for file in files:
        print(os.path.join(root, file))


import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import kaggle_evaluation.jane_street_inference_server
import warnings
import gc
from pathlib import Path

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


# DATASET Exploration
base_path = '/kaggle/input/jane-street-real-time-market-data-forecasting'

print("\nAvailable data files:")
for dirname, _, filenames in os.walk(base_path):
    level = dirname.replace(base_path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(dirname)}/')
    subindent = ' ' * 2 * (level + 1)
    for filename in filenames[:3]:  # Shows first 3 files per directory
        print(f'{subindent}{filename}')
    if len(filenames) > 3:
        print(f'{subindent}... and {len(filenames) - 3} more files')

# Metadata files
print("\n" + "-" * 80)
print("Loading metadata files:")
print("-" * 80)

features_df = pd.read_csv(f'{base_path}/features.csv')
responders_df = pd.read_csv(f'{base_path}/responders.csv')

print(f"\nFeatures metadata: {features_df.shape[0]} features")
print(features_df.head())

print(f"\nResponders metadata: {responders_df.shape[0]} responders")
print(responders_df.head())


# Loaded a sample partition to understand the data structure
print("\nLoading partition 0 for initial exploration:")
sample_df = pd.read_parquet(f'{base_path}/train.parquet/partition_id=0/part-0.parquet')

print(f"\nDataset shape: {sample_df.shape}")
print(f"Memory usage: {sample_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nColumn types:")
print(sample_df.dtypes.value_counts())

print("\nFirst few rows:")
print(sample_df.head())

print("\nBasic statistics:")
print(sample_df.describe())


# Understanding the target variable
print("\nTarget variable: responder_6")
print(f"Mean: {sample_df['responder_6'].mean():.6f}")
print(f"Std: {sample_df['responder_6'].std():.6f}")
print(f"Min: {sample_df['responder_6'].min():.6f}")
print(f"Max: {sample_df['responder_6'].max():.6f}")

# Missing value analysis
print("\n" + "-" * 80)
print("Missing Value Analysis")
print("-" * 80)

feature_cols = [col for col in sample_df.columns if col.startswith('feature_')]
missing_pct = (sample_df[feature_cols].isnull().sum() / len(sample_df) * 100).sort_values(ascending=False)
print(f"\nFeatures with >10% missing values: {(missing_pct > 10).sum()}")
print("\nTop 10 features by missing %:")
print(missing_pct.head(10))

# Time structure
print("\n" + "-" * 80)
print("Time Structure")
print("-" * 80)

print(f"\nUnique date_ids: {sample_df['date_id'].nunique()}")
print(f"Unique time_ids: {sample_df['time_id'].nunique()}")
print(f"Unique symbols: {sample_df['symbol_id'].nunique()}")

print(f"\nRecords per symbol (first 10):")
print(sample_df['symbol_id'].value_counts().head(10))

# Feature distributions
print("\n" + "-" * 80)
print("Feature Distributions")
print("-" * 80)

# Analyzing a few key features
key_features = ['feature_00', 'feature_01', 'feature_02', 'feature_06', 'feature_07']
print("\nKey feature statistics:")
print(sample_df[key_features].describe())


demo_df = sample_df.head(1000).copy()

# Statistical features
available_features = [col for col in feature_cols if col in demo_df.columns]
feature_data = demo_df[available_features]
demo_df['feature_mean'] = feature_data.mean(axis=1)
demo_df['feature_std'] = feature_data.std(axis=1)
demo_df['feature_range'] = feature_data.max(axis=1) - feature_data.min(axis=1)

# Key interactions that has been found important during the training
demo_df['f06_x_time'] = demo_df['feature_06'] * demo_df['time_id']
demo_df['f06_squared'] = demo_df['feature_06'] ** 2
demo_df['time_sin'] = np.sin(2 * np.pi * demo_df['time_id'] / 1000)

print(f"\nOriginal features: {len(available_features)}")
print(f"After engineering: {len([c for c in demo_df.columns if c.startswith('feature_') or c.startswith('f0') or c.startswith('time_')])} feature-related columns")

print("\nNew feature correlations with target:")
new_features = ['feature_mean', 'feature_std', 'f06_x_time', 'f06_squared', 'time_sin']
correlations = demo_df[new_features + ['responder_6']].corr()['responder_6'].drop('responder_6').sort_values(ascending=False)
print(correlations)

# Clean up demo
del demo_df, sample_df
gc.collect()


print("\nTop 10 most important features (from offline training):")
top_features = [
    ('f06_x_time', 'Engineered interaction'),
    ('time_id', 'Original temporal'),
    ('feature_59', 'Original'),
    ('feature_60', 'Original'),
    ('feature_30', 'Original'),
    ('f06_squared', 'Engineered polynomial'),
    ('feature_06', 'Original'),
    ('time_sin', 'Engineered cyclical'),
    ('feature_04', 'Original'),
    ('f07_x_time', 'Engineered interaction')
]

for i, (feat, typ) in enumerate(top_features, 1):
    print(f"  {i:2d}. {feat:20s} ({typ})")


THRESHOLD = 0.0
FEATURE_COLS = [f'feature_{i:02d}' for i in range(79)]

print(f"\nModel configuration:")
print(f"  Decision threshold: {THRESHOLD}")
print(f"  Base features: {len(FEATURE_COLS)}")

print("\nLoading LightGBM model")
try:
    model = lgb.Booster(model_file='/kaggle/input/jane-street-trading-model-v1/models/lgb_model.txt')
    print(f"  Model loaded successfully")
    print(f"  Objective: {model.params.get('objective', 'regression')}")
    print(f"  Number of trees: {model.num_trees()}")
    print(f"  Number of features: {model.num_feature()}")
except Exception as e:
    print(f"✗ Could not load model: {e}")
    print("\nNote: If running without the model, this is just an EDA notebook.")
    print("The model file should be at: /kaggle/input/jane-street-trading-model-v1/models/lgb_model.txt")
    model = None


class InferenceFeatureEngineer:
    """
    Recreates the exact feature engineering from training.

    This class applies the same transformations we used during model training
    to ensure consistency between training and inference.
    """

    def __init__(self):
        self.feature_cols = FEATURE_COLS
        print("\nFeature engineering pipeline:")
        print("   Missing value imputation (fill with 0)")
        print("   Statistical aggregations (6 features)")
        print("   Interaction features (5 features)")
        print("   Polynomial features (4 features)")
        print("   Ratio features (3 features)")
        print("   Deviation features (2 features)")
        print("   Cyclical time encoding (4 features)")
        print(f"  → Total: {len(FEATURE_COLS)} original + 48 engineered = 127 features")

    def transform(self, df):
        """Apply all feature engineering transformations"""
        df_processed = df.copy()

        # Handling missing values
        for col in self.feature_cols:
            if col in df_processed.columns:
                if df_processed[col].isnull().any():
                    df_processed[col].fillna(0, inplace=True)

        # Statistical features
        available_features = [col for col in self.feature_cols if col in df_processed.columns]
        if len(available_features) > 0:
            feature_data = df_processed[available_features]
            df_processed['feature_mean'] = feature_data.mean(axis=1)
            df_processed['feature_std'] = feature_data.std(axis=1).fillna(0)
            df_processed['feature_max'] = feature_data.max(axis=1)
            df_processed['feature_min'] = feature_data.min(axis=1)
            df_processed['feature_range'] = df_processed['feature_max'] - df_processed['feature_min']
            df_processed['feature_count'] = feature_data.notna().sum(axis=1)

        # Interaction features
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

        # Cyclical time encoding
        if 'time_id' in df_processed.columns:
            time_max = 1000
            df_processed['time_normalized'] = df_processed['time_id'] / time_max
            df_processed['time_sin'] = np.sin(2 * np.pi * df_processed['time_normalized'])
            df_processed['time_cos'] = np.cos(2 * np.pi * df_processed['time_normalized'])
            df_processed['time_period'] = (df_processed['time_normalized'] * 3).astype(int).clip(0, 2).astype(float)

        return df_processed

    def prepare_features(self, df):
        """Select and prepare final feature matrix for model"""
        feature_columns = [col for col in df.columns if (
            col.startswith('feature_') or
            col.startswith('f0') or
            col.startswith('time_') or
            col in ['feature_mean', 'feature_std', 'feature_max', 'feature_min',
                   'feature_range', 'feature_count', 'symbol_id', 'time_id', 'weight']
        )]

        feature_columns = list(dict.fromkeys(feature_columns))
        X = df[feature_columns].copy()
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)

        return X

engineer = InferenceFeatureEngineer()
print("\n Feature engineering pipeline is ready")


prediction_count = 0
trade_count = 0
prediction_times = []

def predict(test_df, lags_df):
    """
    Make trading decisions in real-time.

    For each market opportunity:
    1. Engineer features to match training data
    2. Predict expected return
    3. Decide whether to trade
    4. Track performance metrics
    """
    global prediction_count, trade_count, prediction_times

    import time
    start_time = time.time()

    try:
        df_transformed = engineer.transform(test_df)
        X = engineer.prepare_features(df_transformed)
        prediction = model.predict(X.values)[0]
        decision = 1 if prediction > THRESHOLD else 0

        prediction_count += 1
        if decision == 1:
            trade_count += 1

        inference_time = (time.time() - start_time) * 1000
        prediction_times.append(inference_time)

        if prediction_count % 1000 == 0:
            trade_rate = trade_count / prediction_count * 100
            avg_time = np.mean(prediction_times[-1000:])
            print(f"  Predictions: {prediction_count:>6,} | "
                  f"Trades: {trade_count:>6,} ({trade_rate:>5.1f}%) | "
                  f"Latency: {avg_time:>5.2f}ms")

        return pd.DataFrame({'responder_6': [decision]})

    except Exception as e:
        print(f"  Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame({'responder_6': [0]})


if model is not None:
    print("\nInitializing Jane Street inference server")
    print("This will process test data and generate predictions.")
    print("\nMonitoring metrics:")
    print("  - Prediction count: Total opportunities evaluated")
    print("  - Trade count: Decisions to execute trades")
    print("  - Trade rate: % of opportunities we trade")
    print("  - Latency: Average prediction time (should be <20ms)")

    inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

    print("\n" + "-" * 80)
    print("Starting prediction loop")
    print("-" * 80 + "\n")

    inference_server.serve()

    
    # PERFORMANCE ANALYSIS
    
    print("\n" + "=" * 80)
    print("10. PERFORMANCE ANALYSIS")
    print("=" * 80)

    if prediction_count > 0:
        trade_rate = trade_count / prediction_count * 100

        print(f"\nExecution Summary:")
        print(f"  Total predictions: {prediction_count:,}")
        print(f"  Trades executed: {trade_count:,}")
        print(f"  Trade rate: {trade_rate:.1f}%")

        if prediction_times:
            avg_time = np.mean(prediction_times)
            median_time = np.median(prediction_times)
            p95_time = np.percentile(prediction_times, 95)
            p99_time = np.percentile(prediction_times, 99)

            print(f"\nLatency Analysis:")
            print(f"  Average: {avg_time:.2f}ms")
            print(f"  Median: {median_time:.2f}ms")
            print(f"  P95: {p95_time:.2f}ms")
            print(f"  P99: {p99_time:.2f}ms")
            print(f"  Target: <16ms (met: {'Yes' if p95_time < 16 else 'No'})")

        print(f"\nStrategy Assessment:")
        if 40 <= trade_rate <= 60:
            print(f"   GOOD - Conservative strategy")
            print(f"   Trading {trade_rate:.1f}% of opportunities shows good selectivity")
        elif trade_rate > 60:
            print(f"   Too aggressive - Trading {trade_rate:.1f}% of opportunities")
            print(f"   Consider increasing threshold to be more selective")
        else:
            print(f"   Too conservative - Trading only {trade_rate:.1f}% of opportunities")
            print(f"   Consider decreasing threshold to capture more opportunities")

        print(f"\nExpected Performance:")
        print(f"  Based on validation: R² ≈ 0.0137")
        print(f"  Expected on test: R² ≈ 0.012-0.014")
        print(f"  Target leaderboard: Top 15-25%")

    print("\n" + "=" * 80)
    print(" INFERENCE COMPLETE")
    print("=" * 80)

else:
    print("\nNo model loaded - this was an EDA-only run.")
    print("To make predictions, ensure the model file is available at:")
    print("/kaggle/input/jane-street-trading-model-v1/models/lgb_model.txt")

