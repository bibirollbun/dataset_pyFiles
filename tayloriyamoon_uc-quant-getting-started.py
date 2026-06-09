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


#!/usr/bin/env python
# coding: utf-8

# # 25Spr. UC Quant Competition - Complete Working Solution

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import gc
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

# Set random seed
np.random.seed(42)

print("=" * 80)
print("25SPR. UC QUANT COMPETITION - COMPLETE WORKING SOLUTION")
print("=" * 80)

# ## 1. Load All Datasets

print("\n[STEP 1] Loading datasets...")
print("-" * 50)

customers = pd.read_excel('/kaggle/input/25-spr-uc-quant/Customers.xlsx')
suppliers = pd.read_excel('/kaggle/input/25-spr-uc-quant/Suppliers.xlsx')
dataset = pd.read_csv('/kaggle/input/25-spr-uc-quant/dataset.csv')
dataset_extra = pd.read_csv('/kaggle/input/25-spr-uc-quant/dataset_extra.csv')
submission_example = pd.read_csv('/kaggle/input/25-spr-uc-quant/submission_example.csv')

print(f"✓ Main dataset shape: {dataset.shape}")
print(f"✓ Extra dataset shape: {dataset_extra.shape}")
print(f"✓ Customers shape: {customers.shape}")
print(f"✓ Suppliers shape: {suppliers.shape}")
print(f"✓ Submission example shape: {submission_example.shape}")

# Examine submission example format
print("\n[SUBMISSION EXAMPLE]")
print(f"Columns: {list(submission_example.columns)}")
print(f"First 5 rows:")
print(submission_example.head())
print(f"\nData types:")
print(submission_example.dtypes)

# Check target distribution
print(f"\nTarget info:")
print(f"  Non-null targets: {dataset['target'].notna().sum()}")
print(f"  Null targets: {dataset['target'].isna().sum()}")

# ## 2. Create Supply Chain Features

def create_supply_chain_features(df, customers_df, suppliers_df):
    """Create supply chain network features"""
    print("\n[SUPPLY CHAIN FEATURES] Creating features...")
    df = df.copy()
    
    # Initialize counts
    customer_counts = {}
    supplier_counts = {}
    
    # Count connections for each stock
    for col in customers_df.columns[1:]:  # Skip 'Local_Code'
        count = customers_df[col].notna().sum()
        if count > 0:
            customer_counts[col] = count
    
    for col in suppliers_df.columns[1:]:
        count = suppliers_df[col].notna().sum()
        if count > 0:
            supplier_counts[col] = count
    
    # Map to dataframe
    df['n_customers'] = df['S_INFO_WINDCODE'].map(customer_counts).fillna(0)
    df['n_suppliers'] = df['S_INFO_WINDCODE'].map(supplier_counts).fillna(0)
    df['total_connections'] = df['n_customers'] + df['n_suppliers']
    df['customer_supplier_ratio'] = df['n_customers'] / (df['n_suppliers'] + 1)
    
    max_conn = df['total_connections'].max()
    df['supply_chain_importance'] = df['total_connections'] / (max_conn + 1)
    
    print("✓ Created supply chain features")
    return df

# ## 3. Create Time Series Features

def create_time_series_features(df, extra_df):
    """Create time series features efficiently"""
    print("\n[TIME SERIES FEATURES] Processing extra dataset...")
    
    # Calculate basic statistics per stock
    print("Calculating statistics per stock...")
    
    # Group by stock and calculate features
    ts_features = extra_df.groupby('S_INFO_WINDCODE').agg({
        'S_DQ_PCTCHANGE': ['mean', 'std', 'min', 'max', 'count'],
        'S_DQ_VOLUME': ['mean', 'std'],
        'S_DQ_AMOUNT': ['mean', 'std']
    })
    
    # Flatten column names
    ts_features.columns = ['_'.join(col).strip() for col in ts_features.columns.values]
    ts_features = ts_features.reset_index()
    
    # Calculate additional features
    print("Calculating momentum and volatility features...")
    momentum_features = []
    
    for stock in extra_df['S_INFO_WINDCODE'].unique():
        stock_data = extra_df[extra_df['S_INFO_WINDCODE'] == stock].sort_values('TRADE_DT')
        pct_changes = stock_data['S_DQ_PCTCHANGE'].values
        
        features = {'S_INFO_WINDCODE': stock}
        
        # Volatility measures
        if len(pct_changes) >= 2:
            features['pct_range_90'] = np.percentile(pct_changes, 95) - np.percentile(pct_changes, 5)
        else:
            features['pct_range_90'] = 0
            
        if len(pct_changes) > 3:
            features['pct_skew'] = stats.skew(pct_changes)
        else:
            features['pct_skew'] = 0
            
        # Momentum
        features['momentum_5d'] = np.sum(pct_changes[-5:]) if len(pct_changes) >= 5 else 0
        features['momentum_20d'] = np.sum(pct_changes[-20:]) if len(pct_changes) >= 20 else 0
        
        momentum_features.append(features)
    
    momentum_df = pd.DataFrame(momentum_features)
    
    # Merge all time series features
    ts_features = ts_features.merge(momentum_df, on='S_INFO_WINDCODE', how='left')
    
    # Merge with main dataframe
    df = df.merge(ts_features, on='S_INFO_WINDCODE', how='left')
    
    # Fill missing values
    ts_cols = [col for col in ts_features.columns if col != 'S_INFO_WINDCODE']
    for col in ts_cols:
        df[col] = df[col].fillna(0)
    
    print(f"✓ Added {len(ts_cols)} time series features")
    return df

# Apply features
dataset = create_supply_chain_features(dataset, customers, suppliers)
dataset = create_time_series_features(dataset, dataset_extra)

# Clean up memory
del customers, suppliers, dataset_extra
gc.collect()

# ## 4. Feature Engineering

print("\n[FEATURE ENGINEERING] Creating derived features...")

# Small epsilon to avoid division by zero
eps = 1e-10

# Flow features
total_flow = (np.abs(dataset['Net_Flow_ExLarge']) + np.abs(dataset['Net_Flow_Large']) + 
              np.abs(dataset['Net_Flow_Med']) + np.abs(dataset['Net_Flow_Small']) + eps)

dataset['large_flow_dominance'] = (dataset['Net_Flow_ExLarge'] + dataset['Net_Flow_Large']) / total_flow
dataset['flow_imbalance'] = ((dataset['Net_Flow_ExLarge'] + dataset['Net_Flow_Large']) - 
                             (dataset['Net_Flow_Med'] + dataset['Net_Flow_Small'])) / total_flow

# Risk metrics
dataset['sharpe_ratio'] = dataset['Return_1M_Past'] / (dataset['Volatility'] + eps)
dataset['volatility_scaled'] = dataset['Volatility'] / (dataset['S_DQ_PCTCHANGE_std'] + eps)

# Fundamental ratios
dataset['earnings_yield'] = dataset['EST_EPS'] / (np.abs(dataset['NET_PROFIT']) + 1)
dataset['profit_margin'] = dataset['NET_PROFIT'] / (np.abs(dataset['EST_OPER_PROFIT']) + 1)
dataset['roe_stability'] = dataset['EST_ROE'] / (dataset['S_DQ_PCTCHANGE_std'] + eps)

# Network-weighted features
dataset['network_roe'] = dataset['EST_ROE'] * dataset['supply_chain_importance']
dataset['network_eps'] = dataset['EST_EPS'] * dataset['supply_chain_importance']

# Transformations
dataset['log_volatility'] = np.log1p(np.abs(dataset['Volatility']))
dataset['sqrt_connections'] = np.sqrt(dataset['total_connections'])

print("✓ Created derived features")

# ## 5. Handle Missing Values

print("\n[MISSING VALUES] Handling missing values...")

# Get feature columns
numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in ['target', 'TRADE_DT']]

print(f"Number of features: {len(feature_cols)}")

# Fill missing values
for col in feature_cols:
    if dataset[col].isnull().any():
        if col in ['n_customers', 'n_suppliers', 'total_connections']:
            dataset[col] = dataset[col].fillna(0)
        else:
            median_val = dataset[col].median()
            dataset[col] = dataset[col].fillna(median_val if not pd.isna(median_val) else 0)

print("✓ Missing values handled")

# ## 6. Model Training with Cross-Validation

print("\n[MODEL TRAINING] Training models with cross-validation...")

# Since all data has targets, we'll use cross-validation
X = dataset[feature_cols].values
y = dataset['target'].values

print(f"Data shape: X={X.shape}, y={y.shape}")

# Feature selection
print("\nSelecting best features...")
k_features = min(40, len(feature_cols))
selector = SelectKBest(f_regression, k=k_features)
X_selected = selector.fit_transform(X, y)
selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]

print(f"Selected {len(selected_features)} features")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Define models
models = {
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'RandomForest': RandomForestRegressor(
        n_estimators=50,  # Reduced for speed
        max_depth=10,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': xgb.XGBRegressor(
        n_estimators=50,  # Reduced for speed
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ),
    'LightGBM': lgb.LGBMRegressor(
        n_estimators=50,  # Reduced for speed
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
}

# Cross-validation
print("\nPerforming 3-fold cross-validation...")
kf = KFold(n_splits=3, shuffle=True, random_state=42)
cv_scores = {}

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        fold_scores.append(mse)
        
        print(f"  Fold {fold+1}: MSE = {mse:.6f}")
    
    cv_scores[name] = np.mean(fold_scores)
    print(f"  Average MSE: {cv_scores[name]:.6f}")

# Calculate ensemble weights
mse_values = list(cv_scores.values())
weights = 1 / np.array(mse_values)
weights = weights / weights.sum()

print("\nEnsemble weights:")
for name, weight in zip(models.keys(), weights):
    print(f"  {name}: {weight:.4f}")

# ## 7. Train Final Models

print("\n[FINAL TRAINING] Training models on full dataset...")

final_models = {}
final_predictions = np.zeros(len(X_scaled))

for name, model in models.items():
    print(f"Training final {name}...")
    
    # Create fresh model instance
    if name == 'Ridge':
        final_model = Ridge(alpha=1.0, random_state=42)
    elif name == 'RandomForest':
        final_model = RandomForestRegressor(
            n_estimators=50, max_depth=10, min_samples_split=20,
            random_state=42, n_jobs=-1
        )
    elif name == 'XGBoost':
        final_model = xgb.XGBRegressor(
            n_estimators=50, learning_rate=0.1, max_depth=6,
            subsample=0.8, random_state=42, n_jobs=-1, verbosity=0
        )
    else:  # LightGBM
        final_model = lgb.LGBMRegressor(
            n_estimators=50, learning_rate=0.1, num_leaves=31,
            subsample=0.8, random_state=42, n_jobs=-1, verbosity=-1
        )
    
    # Train on full data
    final_model.fit(X_scaled, y)
    final_models[name] = final_model
    
    # Add weighted predictions
    predictions = final_model.predict(X_scaled)
    final_predictions += weights[list(models.keys()).index(name)] * predictions

# Add predictions to dataset
dataset['prediction'] = final_predictions

# ## 8. Create Submission

print("\n[SUBMISSION] Creating submission file...")

# Get unique stocks
unique_stocks = sorted(dataset['S_INFO_WINDCODE'].unique())
print(f"Total unique stocks: {len(unique_stocks)}")

# Create mapping for first 1800 stocks
stock_to_id = {stock: idx + 1 for idx, stock in enumerate(unique_stocks[:1800])}

# Aggregate predictions by stock
print("Aggregating predictions by stock...")
stock_predictions = dataset.groupby('S_INFO_WINDCODE').agg({
    'prediction': ['mean', 'median', 'std', 'count'],
    'target': 'mean'  # Also get actual target mean for comparison
}).reset_index()

# Flatten columns
stock_predictions.columns = ['S_INFO_WINDCODE', 'pred_mean', 'pred_median', 
                             'pred_std', 'pred_count', 'target_mean']

# Add stock IDs
stock_predictions['stock_id'] = stock_predictions['S_INFO_WINDCODE'].map(stock_to_id)

# Filter to stocks with IDs
stock_predictions_valid = stock_predictions[stock_predictions['stock_id'].notna()].copy()

print(f"Stocks with valid IDs: {len(stock_predictions_valid)}")

# Create submission
submission = pd.DataFrame({
    'S_INFO_WINDCODE': stock_predictions_valid['stock_id'].astype(int),
    'target': stock_predictions_valid['pred_mean']
})

# Handle missing stocks
if len(submission) < 1800:
    missing_ids = set(range(1, 1801)) - set(submission['S_INFO_WINDCODE'])
    
    # Use global mean as default
    global_mean = dataset['target'].mean()
    print(f"\nAdding {len(missing_ids)} missing stocks with global mean: {global_mean:.6f}")
    
    missing_df = pd.DataFrame({
        'S_INFO_WINDCODE': sorted(list(missing_ids)),
        'target': global_mean
    })
    
    submission = pd.concat([submission, missing_df], ignore_index=True)

# Sort by stock ID first
submission = submission.sort_values('S_INFO_WINDCODE').reset_index(drop=True)

# Add rank column based on target values
# IMPORTANT: Assuming higher predicted target = better stock = lower rank (1 is best)
# If the competition expects opposite ranking, change ascending=False to ascending=True
submission['rank'] = submission['target'].rank(method='first', ascending=False).astype(int)

# Alternative ranking approaches (commented out):
# For percentile ranking: submission['rank'] = submission['target'].rank(pct=True)
# For dense ranking: submission['rank'] = submission['target'].rank(method='dense', ascending=False)
# For average ranking: submission['rank'] = submission['target'].rank(method='average', ascending=False)

# Verify rank column
print(f"\nRank statistics:")
print(f"  Min rank: {submission['rank'].min()}")
print(f"  Max rank: {submission['rank'].max()}")
print(f"  Unique ranks: {submission['rank'].nunique()}")

# Double-check: show stocks with highest and lowest predicted targets
print("\nStocks with highest predicted targets (should have lowest ranks):")
print(submission.nlargest(5, 'target')[['S_INFO_WINDCODE', 'target', 'rank']])

print("\nStocks with lowest predicted targets (should have highest ranks):")
print(submission.nsmallest(5, 'target')[['S_INFO_WINDCODE', 'target', 'rank']])

# Save submission
# Ensure columns are in the correct order
submission = submission[['S_INFO_WINDCODE', 'target', 'rank']]

# Final sort by S_INFO_WINDCODE to ensure order
submission = submission.sort_values('S_INFO_WINDCODE').reset_index(drop=True)

submission.to_csv('submission.csv', index=False)

print("\n✅ Submission saved with columns:", list(submission.columns))

# ## 9. Summary and Validation

print("\n[VALIDATION] Submission validation...")
print(f"Submission shape: {submission.shape}")
print(f"Submission columns: {list(submission.columns)}")
print(f"Unique stock IDs: {submission['S_INFO_WINDCODE'].nunique()}")
print(f"Stock ID range: {submission['S_INFO_WINDCODE'].min()} to {submission['S_INFO_WINDCODE'].max()}")

# Check for duplicates
duplicates = submission['S_INFO_WINDCODE'].duplicated().sum()
print(f"Duplicate stock IDs: {duplicates}")

# Statistics
print(f"\nTarget statistics:")
print(f"  Mean: {submission['target'].mean():.6f}")
print(f"  Std: {submission['target'].std():.6f}")
print(f"  Min: {submission['target'].min():.6f}")
print(f"  Max: {submission['target'].max():.6f}")

print(f"\nRank statistics:")
print(f"  Min: {submission['rank'].min()}")
print(f"  Max: {submission['rank'].max()}")
print(f"  Unique ranks: {submission['rank'].nunique()}")

# Show top ranked stocks
print("\nTop 10 ranked stocks (best predictions):")
top_10 = submission.nsmallest(10, 'rank')[['S_INFO_WINDCODE', 'target', 'rank']]
print(top_10)

# Show bottom ranked stocks
print("\nBottom 10 ranked stocks (worst predictions):")
bottom_10 = submission.nlargest(10, 'rank')[['S_INFO_WINDCODE', 'target', 'rank']]
print(bottom_10)

print("\nFirst 10 rows of submission:")
print(submission.head(10))

# Compare with training target distribution
print("\nComparison with training data:")
print(f"  Training target mean: {dataset['target'].mean():.6f}")
print(f"  Submission target mean: {submission['target'].mean():.6f}")

# ## 10. Feature Importance (Optional)

print("\n[FEATURE IMPORTANCE] Top features from Random Forest:")
rf_model = final_models['RandomForest']
importances = rf_model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 15 most important features:")
for idx, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

print("\n[FINAL VALIDATION]")
print("Required columns check:")
required_cols = ['S_INFO_WINDCODE', 'target', 'rank']
for col in required_cols:
    if col in submission.columns:
        print(f"  ✅ {col}: present")
    else:
        print(f"  ❌ {col}: MISSING!")

print("\nData types:")
print(submission.dtypes)

print("\nSubmission shape:", submission.shape)
print("Expected shape: (1800, 3)")

# Verify all ranks are unique and in correct range
assert submission['rank'].min() == 1, "Minimum rank should be 1"
assert submission['rank'].max() == 1800, "Maximum rank should be 1800"
assert submission['rank'].nunique() == 1800, "All ranks should be unique"
print("✅ All rank validations passed!")

print("\n" + "=" * 80)
print("COMPLETE! Submission saved to 'submission.csv'")
print("=" * 80)

