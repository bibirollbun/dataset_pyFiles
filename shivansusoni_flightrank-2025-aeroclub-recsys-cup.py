#!/usr/bin/env python3
"""
Business Flight Recommendation Model - Fast Optimized Version
Key optimizations:
1. Vectorized operations instead of apply()
2. Simplified model with fewer estimators
3. Efficient feature engineering
4. Reduced cross-validation folds
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
N_FOLDS = 2  # Reduced for speed
N_ESTIMATORS = 50  # Reduced for speed

def parse_duration_vectorized(duration_series):
    """Vectorized duration parsing - much faster than apply()"""
    # Convert to string type for consistent handling
    duration_str = duration_series.astype(str)
    
    # Create output array
    result = np.full(len(duration_series), np.nan)
    
    # Handle numeric values that were converted to string
    numeric_mask = duration_str.str.match(r'^\d+\.?\d*$')
    result[numeric_mask] = pd.to_numeric(duration_str[numeric_mask])
    
    # Handle HH:MM:SS format
    time_mask = duration_str.str.contains(':', na=False)
    if time_mask.any():
        time_parts = duration_str[time_mask].str.split(':', expand=True)
        if time_parts.shape[1] >= 2:
            hours = pd.to_numeric(time_parts[0], errors='coerce')
            minutes = pd.to_numeric(time_parts[1], errors='coerce')
            result[time_mask] = hours * 60 + minutes
    
    return result

def load_and_prepare_data():
    """Load data with optimized duration parsing"""
    print("Loading data...")
    train = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet')
    test = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    # Fast duration parsing
    for df in [train, test]:
        if 'legs0_duration' in df.columns:
            print("Parsing durations (optimized)...")
            df['duration_minutes'] = parse_duration_vectorized(df['legs0_duration'])
            # Fast median fill
            median_val = np.nanmedian(df['duration_minutes'])
            df['duration_minutes'].fillna(median_val, inplace=True)
        else:
            df['duration_minutes'] = 180  # Default 3 hours
    
    return train, test

def create_features_fast(df):
    """Optimized feature creation with minimal features for speed"""
    print("Creating features (fast mode)...")
    
    # Essential price features only
    df['tax_rate'] = df['taxes'] / (df['totalPrice'] + 1e-5)
    
    # Fast group statistics using transform (avoids merge)
    df['price_min'] = df.groupby('ranker_id')['totalPrice'].transform('min')
    df['price_mean'] = df.groupby('ranker_id')['totalPrice'].transform('mean')
    df['duration_min'] = df.groupby('ranker_id')['duration_minutes'].transform('min')
    
    # Core relative features
    df['price_ratio'] = df['totalPrice'] / (df['price_min'] + 1e-5)
    df['duration_ratio'] = df['duration_minutes'] / (df['duration_min'] + 1e-5)
    df['is_cheapest'] = (df['totalPrice'] == df['price_min']).astype(int)
    df['is_fastest'] = (df['duration_minutes'] == df['duration_min']).astype(int)
    
    # Simple rankings
    df['price_rank'] = df.groupby('ranker_id')['totalPrice'].rank(method='min')
    df['duration_rank'] = df.groupby('ranker_id')['duration_minutes'].rank(method='min')
    df['combined_rank'] = df['price_rank'] + df['duration_rank']
    
    # Direct flight check (simplified)
    segment_cols = [col for col in df.columns if 'segments1_departureFrom' in col]
    df['is_direct'] = segment_cols[0] if segment_cols else 1
    df['is_direct'] = df['is_direct'].isna().astype(int)
    
    # Time features (simplified)
    df['request_hour'] = pd.to_datetime(df['requestDate']).dt.hour
    
    # Policy compliance
    if 'pricingInfo_isAccessTP' in df.columns:
        df['policy_compliant'] = df['pricingInfo_isAccessTP'].fillna(0).astype(int)
    else:
        df['policy_compliant'] = 1
    
    # Group size (useful for model)
    df['group_size'] = df.groupby('ranker_id')['ranker_id'].transform('count')
    
    return df

def get_essential_features():
    """Return minimal feature set for speed"""
    return [
        'totalPrice', 'tax_rate',
        'price_ratio', 'is_cheapest', 'price_rank',
        'duration_minutes', 'duration_ratio', 'is_fastest', 'duration_rank',
        'combined_rank', 'is_direct',
        'request_hour', 'policy_compliant', 'group_size'
    ]

def train_fast_model(train_df, feature_cols):
    """Train faster model with RandomForest"""
    print(f"\nTraining fast model with {N_FOLDS}-fold CV...")
    
    X = train_df[feature_cols].fillna(0)
    y = train_df['selected']
    groups = train_df['ranker_id']
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    models = []
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"Fold {fold + 1}/{N_FOLDS}...")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # RandomForest is faster than GradientBoosting
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            n_jobs=-1,  # Use all cores
            random_state=RANDOM_STATE + fold,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        models.append(model)
        
        # Quick validation
        val_pred = model.predict_proba(X_val)[:, 1]
        val_df = train_df.iloc[val_idx][['ranker_id', 'selected']].copy()
        val_df['score'] = val_pred
        
        # Fast metric calculation
        hit_rate = calculate_hitrate3_fast(val_df)
        scores.append(hit_rate)
        print(f"  HitRate@3: {hit_rate:.4f}")
    
    print(f"\nAverage HitRate@3: {np.mean(scores):.4f}")
    return models

def calculate_hitrate3_fast(df):
    """Optimized HitRate@3 calculation"""
    # Filter groups with >10 options
    group_sizes = df.groupby('ranker_id').size()
    valid_groups = group_sizes[group_sizes > 10].index
    df_filtered = df[df['ranker_id'].isin(valid_groups)]
    
    if len(df_filtered) == 0:
        return 0
    
    # Get top 3 for each group
    top3_idx = df_filtered.groupby('ranker_id')['score'].nlargest(3).index.get_level_values(1)
    top3_selected = df_filtered.loc[top3_idx, 'selected'].groupby(
        df_filtered.loc[top3_idx, 'ranker_id']
    ).sum()
    
    hits = (top3_selected > 0).sum()
    total = len(valid_groups)
    
    return hits / total if total > 0 else 0

def create_submission_fast(test_df, models, feature_cols):
    """Fast submission creation"""
    print("\nCreating submission...")
    
    X_test = test_df[feature_cols].fillna(0)
    
    # Fast ensemble prediction
    test_scores = np.mean([
        model.predict_proba(X_test)[:, 1] for model in models
    ], axis=0)
    
    # Create submission efficiently
    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'ranker_id': test_df['ranker_id'],
        'score': test_scores
    })
    
    # Fast ranking
    submission['selected'] = submission.groupby('ranker_id')['score'].rank(
        method='first',
        ascending=False
    ).astype(int)
    
    return submission[['Id', 'ranker_id', 'selected']]

def main():
    """Main execution - optimized for speed"""
    import time
    start_time = time.time()
    
    # Load data
    train, test = load_and_prepare_data()
    
    # Create features
    train = create_features_fast(train)
    test = create_features_fast(test)
    
    # Get features
    feature_cols = get_essential_features()
    feature_cols = [col for col in feature_cols if col in train.columns and col in test.columns]
    print(f"\nUsing {len(feature_cols)} features for speed")
    
    # Train model
    models = train_fast_model(train, feature_cols)
    
    # Create submission
    submission = create_submission_fast(test, models, feature_cols)
    
    # Quick validation
    print("\nValidating submission...")
    assert len(submission) == len(test), "Submission length mismatch"
    
    # Save
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission saved to submission.csv")
    print(submission.head())
    
    # Time taken
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds")
    
    # Top features
    if hasattr(models[0], 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': np.mean([m.feature_importances_ for m in models], axis=0)
        }).sort_values('importance', ascending=False)
        print("\nTop 10 Features:")
        print(importance.head(10))

if __name__ == "__main__":
    main()

