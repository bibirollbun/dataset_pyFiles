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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scipy.stats import pearsonr, spearmanr, rankdata
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class Config:
    """Configuration parameters for robust prediction pipeline"""
    # Paths
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    sample_sub_path = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    # Data settings
    use_recent_months = 9  
    n_top_features = 200  # Increased for better coverage
    n_folds = 5
    random_seed = 42
    validation_gap_days = 7  # Gap between train and validation
    
    # Feature engineering parameters
    correlation_threshold = 0.90  # Less aggressive removal
    max_feature_interactions = 150
    stability_check_periods = 4
    
    # Model parameters
    use_ensemble_optimization = True
    max_model_weight = 0.5  # Cap for individual model in ensemble

def load_and_prepare_data():
    """Load and prepare data with temporal awareness"""
    print("Loading data...")
    train = pd.read_parquet(Config.train_path)
    test = pd.read_parquet(Config.test_path)
    
    # Sort by timestamp for proper temporal handling
    if 'timestamp' in train.columns:
        train['timestamp'] = pd.to_datetime(train['timestamp'])
        train = train.sort_values('timestamp').reset_index(drop=True)
        
        # Use recent data but keep enough for validation
        if Config.use_recent_months:
            cutoff = train['timestamp'].max() - pd.DateOffset(months=Config.use_recent_months)
            train = train[train['timestamp'] >= cutoff].reset_index(drop=True)
            print(f"Using data from last {Config.use_recent_months} months: {len(train)} rows")
    
    return train, test

def create_time_series_splits(train_df, n_splits=5):
    """Create proper time series validation splits with gaps"""
    if 'timestamp' not in train_df.columns:
        # Fallback to position-based splitting
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=len(train_df)//10)
        return list(tscv.split(train_df))
    
    splits = []
    total_days = (train_df['timestamp'].max() - train_df['timestamp'].min()).days
    
    for i in range(n_splits):
        # Progressive training windows
        train_ratio = 0.4 + (0.1 * i)  # 40% to 80% for training
        val_ratio = 0.1  # 10% for validation
        
        train_days = int(total_days * train_ratio)
        val_days = int(total_days * val_ratio)
        
        train_end = train_df['timestamp'].min() + pd.Timedelta(days=train_days)
        val_start = train_end + pd.Timedelta(days=Config.validation_gap_days)
        val_end = val_start + pd.Timedelta(days=val_days)
        
        train_idx = train_df[train_df['timestamp'] <= train_end].index.tolist()
        val_idx = train_df[(train_df['timestamp'] > val_start) & 
                          (train_df['timestamp'] <= val_end)].index.tolist()
        
        if len(train_idx) > 1000 and len(val_idx) > 100:
            splits.append((train_idx, val_idx))
            print(f"Split {i+1}: Train {len(train_idx)}, Val {len(val_idx)}")
    
    return splits

def robust_standardize(series):
    """Robust standardization using median and MAD"""
    median = series.median()
    mad = (series - median).abs().median()
    return (series - median) / (mad * 1.4826 + 1e-6)

def create_market_microstructure_features(df):
    """Create domain-specific market microstructure features"""
    features = []
    
    # Order book imbalance with multiple time horizons
    if all(col in df.columns for col in ['bid_qty', 'ask_qty']):
        # Basic imbalance
        df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
        df['bid_ask_ratio'] = df['bid_qty'] / (df['ask_qty'] + 1e-10)
        df['total_depth'] = df['bid_qty'] + df['ask_qty']
        features.extend(['bid_ask_imbalance', 'bid_ask_ratio', 'total_depth'])
        
        # Quote stability metrics
        for window in [5, 10, 30]:
            df[f'quote_stability_{window}'] = 1 / (1 + 
                df['bid_qty'].rolling(window, min_periods=1).std() + 
                df['ask_qty'].rolling(window, min_periods=1).std() + 1e-10)
            features.append(f'quote_stability_{window}')
    
    # Order flow analysis
    if all(col in df.columns for col in ['buy_qty', 'sell_qty']):
        df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)
        df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
        df['order_flow_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-10)
        features.extend(['order_flow_imbalance', 'net_order_flow', 'order_flow_ratio'])
        
        # Order flow momentum
        for window in [10, 30, 60]:
            df[f'order_flow_momentum_{window}'] = df['net_order_flow'].rolling(window, min_periods=1).mean()
            features.append(f'order_flow_momentum_{window}')
    
    # Volume analysis
    if 'volume' in df.columns:
        # Volume regimes
        df['volume_percentile'] = df['volume'].rolling(1440, min_periods=60).rank(pct=True)
        df['volume_regime'] = pd.cut(df['volume_percentile'], bins=5, labels=False, duplicates='drop')
        df['volume_zscore'] = (df['volume'] - df['volume'].rolling(60, min_periods=1).mean()) / (
            df['volume'].rolling(60, min_periods=1).std() + 1e-10)
        features.extend(['volume_percentile', 'volume_regime', 'volume_zscore'])
        
        # Volume-weighted features
        if 'bid_qty' in df.columns:
            df['bid_volume_weighted'] = df['bid_qty'] * df['volume'] / (
                df['volume'].rolling(60, min_periods=1).mean() + 1e-10)
            features.append('bid_volume_weighted')
        
        if 'ask_qty' in df.columns:
            df['ask_volume_weighted'] = df['ask_qty'] * df['volume'] / (
                df['volume'].rolling(60, min_periods=1).mean() + 1e-10)
            features.append('ask_volume_weighted')
    
    # Market pressure indicators
    if all(col in df.columns for col in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty']):
        df['buy_pressure'] = df['buy_qty'] / (df['ask_qty'] + 1e-10)
        df['sell_pressure'] = df['sell_qty'] / (df['bid_qty'] + 1e-10)
        df['net_pressure'] = df['buy_pressure'] - df['sell_pressure']
        features.extend(['buy_pressure', 'sell_pressure', 'net_pressure'])
    
    return features

def create_rolling_features_with_stability(df, base_cols):
    """Create rolling features with focus on stability"""
    features = []
    
    # Define multiple time scales
    windows = [5, 10, 20, 50, 100]
    
    for col in base_cols:
        if col not in df.columns:
            continue
            
        for window in windows:
            # Basic rolling statistics
            roll_mean = df[col].shift(1).rolling(window, min_periods=max(1, window//5)).mean()
            roll_std = df[col].shift(1).rolling(window, min_periods=max(1, window//5)).std()
            
            df[f'{col}_roll_mean_{window}'] = roll_mean
            df[f'{col}_roll_std_{window}'] = roll_std
            df[f'{col}_roll_zscore_{window}'] = (df[col] - roll_mean) / (roll_std + 1e-10)
            
            # Robust statistics
            roll_median = df[col].shift(1).rolling(window, min_periods=max(1, window//5)).median()
            roll_mad = (df[col].shift(1).rolling(window, min_periods=max(1, window//5))
                       .apply(lambda x: np.median(np.abs(x - np.median(x)))))
            
            df[f'{col}_roll_median_{window}'] = roll_median
            df[f'{col}_roll_mad_{window}'] = roll_mad
            df[f'{col}_roll_robust_zscore_{window}'] = (df[col] - roll_median) / (roll_mad * 1.4826 + 1e-10)
            
            features.extend([
                f'{col}_roll_mean_{window}',
                f'{col}_roll_std_{window}',
                f'{col}_roll_zscore_{window}',
                f'{col}_roll_median_{window}',
                f'{col}_roll_mad_{window}',
                f'{col}_roll_robust_zscore_{window}'
            ])
    
    return features

def create_stable_feature_interactions(df, base_features, target_col='label'):
    """Create feature interactions based on stability across time periods"""
    features = []
    
    # Split data into periods for stability assessment
    n_periods = Config.stability_check_periods
    period_size = len(df) // n_periods
    
    # Calculate feature importance stability
    feature_stability_scores = {}
    
    for feat in base_features[:100]:  # Limit to top features
        if feat not in df.columns:
            continue
            
        period_correlations = []
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else len(df)
            
            if target_col in df.columns:
                period_corr = abs(spearmanr(
                    df[feat].iloc[start_idx:end_idx].fillna(0),
                    df[target_col].iloc[start_idx:end_idx]
                )[0])
                period_correlations.append(period_corr)
        
        if period_correlations:
            mean_corr = np.mean(period_correlations)
            std_corr = np.std(period_correlations)
            stability_score = mean_corr / (1 + std_corr)
            feature_stability_scores[feat] = stability_score
    
    # Select stable features for interactions
    stable_features = sorted(feature_stability_scores.items(), 
                           key=lambda x: x[1], reverse=True)[:30]
    
    # Create interactions only between stable features
    interaction_count = 0
    for i, (feat1, score1) in enumerate(stable_features):
        for feat2, score2 in stable_features[i+1:i+5]:  # Limited interactions
            if interaction_count >= Config.max_feature_interactions:
                break
                
            if score1 * score2 > 0.001:  # Both features must be reasonably stable
                # Multiplication interaction
                interaction_name = f'interact_{feat1}_{feat2}'
                df[interaction_name] = robust_standardize(df[feat1]) * robust_standardize(df[feat2])
                features.append(interaction_name)
                
                # Difference interaction
                diff_name = f'diff_{feat1}_{feat2}'
                df[diff_name] = robust_standardize(df[feat1]) - robust_standardize(df[feat2])
                features.append(diff_name)
                
                interaction_count += 2
    
    print(f"Created {len(features)} stable feature interactions")
    return features

def create_all_features(df, is_train=True):
    """Comprehensive feature engineering with stability focus"""
    features = []
    
    # Start with base features
    base_features = [col for col in df.columns if col not in ['timestamp', 'label']]
    features.extend(base_features)
    
    # Add market microstructure features
    print("Creating market microstructure features...")
    market_features = create_market_microstructure_features(df)
    features.extend(market_features)
    
    # Add rolling features for key columns
    print("Creating rolling features...")
    key_cols = ['volume', 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'net_order_flow']
    key_cols = [col for col in key_cols if col in df.columns][:5]
    rolling_features = create_rolling_features_with_stability(df, key_cols)
    features.extend(rolling_features)
    
    # Add stable feature interactions
    if is_train:
        print("Creating stable feature interactions...")
        interaction_features = create_stable_feature_interactions(df, base_features)
        features.extend(interaction_features)
    
    # Clean features
    for col in features:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(0)
    
    # Remove duplicates
    features = list(dict.fromkeys(features))
    print(f"Total features created: {len(features)}")
    
    return df, features

def select_stable_features(train_df, features, n_features=200):
    """Select features based on stability across time periods"""
    print(f"\nSelecting {n_features} stable features from {len(features)}...")
    
    # Create time-based splits for stability assessment
    splits = create_time_series_splits(train_df, n_splits=Config.stability_check_periods)
    
    feature_scores = {feat: [] for feat in features}
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"Evaluating features on fold {fold + 1}...")
        
        X_train = train_df.iloc[train_idx][features]
        y_train = train_df.iloc[train_idx]['label']
        
        # Use RandomForest for feature importance
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=50,
            random_state=Config.random_seed + fold
        )
        rf.fit(X_train, y_train)
        
        # Store importance scores
        for feat, importance in zip(features, rf.feature_importances_):
            feature_scores[feat].append(importance)
    
    # Calculate stability metrics
    feature_stability = {}
    for feat, scores in feature_scores.items():
        if scores:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            stability = mean_score / (1 + std_score)
            feature_stability[feat] = stability
    
    # Remove highly correlated features among top candidates
    top_candidates = sorted(feature_stability.items(), 
                          key=lambda x: x[1], reverse=True)[:int(n_features * 1.5)]
    
    # Calculate correlation matrix for top candidates
    candidate_features = [feat for feat, _ in top_candidates]
    corr_matrix = train_df[candidate_features].corr().abs()
    
    # Remove correlated features
    selected_features = []
    for feat, score in top_candidates:
        if len(selected_features) >= n_features:
            break
            
        # Check correlation with already selected features
        is_correlated = False
        for selected_feat in selected_features:
            if feat in corr_matrix.columns and selected_feat in corr_matrix.index:
                if corr_matrix.loc[selected_feat, feat] > Config.correlation_threshold:
                    is_correlated = True
                    break
        
        if not is_correlated:
            selected_features.append(feat)
    
    print(f"\nTop 10 stable features:")
    for i, feat in enumerate(selected_features[:10]):
        print(f"  {feat}: stability={feature_stability[feat]:.4f}")
    
    return selected_features

def get_regularized_models():
    """Get models with strong regularization"""
    models = {
        'xgb_regularized': XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.6,
            colsample_bytree=0.6,
            reg_alpha=10,
            reg_lambda=10,
            min_child_weight=10,
            gamma=0.5,
            random_state=Config.random_seed
        ),
        
        'lgb_regularized': LGBMRegressor(
            n_estimators=400,
            num_leaves=31,
            max_depth=5,
            learning_rate=0.01,
            feature_fraction=0.6,
            bagging_fraction=0.6,
            bagging_freq=1,
            reg_alpha=10,
            reg_lambda=10,
            min_data_in_leaf=50,
            min_gain_to_split=0.1,
            random_state=Config.random_seed,
            verbose=-1
        ),
        
        'xgb_conservative': XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.005,
            subsample=0.5,
            colsample_bytree=0.5,
            reg_alpha=20,
            reg_lambda=20,
            min_child_weight=20,
            gamma=1.0,
            random_state=Config.random_seed + 1
        ),
        
        'ridge_robust': Ridge(
            alpha=10.0,
            random_state=Config.random_seed
        ),
        
        'elastic_robust': ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            random_state=Config.random_seed
        )
    }
    
    return models

def train_with_time_series_validation(train_df, test_df, features):
    """Train models using proper time series validation"""
    X_train = train_df[features]
    y_train = train_df['label']
    X_test = test_df[features]
    
    # Remove NaN rows
    valid_idx = ~(X_train.isna().any(axis=1) | y_train.isna())
    X_train = X_train[valid_idx]
    y_train = y_train[valid_idx]
    
    # Use RobustScaler for outlier resistance
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    print(f"\nTraining on {len(X_train)} samples with {len(features)} features")
    
    models = get_regularized_models()
    all_predictions = {}
    model_scores = {}
    oof_predictions = {}
    
    # Create time series splits
    splits = create_time_series_splits(train_df[valid_idx].reset_index(drop=True), 
                                     n_splits=Config.n_folds)
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        fold_predictions = []
        fold_scores = []
        oof_pred = np.zeros(len(X_train))
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_fold_train = X_train_scaled.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train_scaled.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # Time-decay weights for training samples
            train_size = len(X_fold_train)
            time_weights = np.linspace(0.5, 1.0, train_size)
            
            # Clone and train model
            model_clone = model.__class__(**model.get_params())
            
            if 'xgb' in model_name:
                model_clone.fit(
                    X_fold_train, y_fold_train,
                    sample_weight=time_weights,
                    eval_set=[(X_fold_val, y_fold_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            elif 'lgb' in model_name:
                import lightgbm as lgbm
                model_clone.fit(
                    X_fold_train, y_fold_train,
                    sample_weight=time_weights,
                    eval_set=[(X_fold_val, y_fold_val)],
                    callbacks=[lgbm.early_stopping(50, verbose=False)]
                )
            elif 'cat' in model_name:
                model_clone.fit(
                    X_fold_train, y_fold_train,
                    sample_weight=time_weights,
                    eval_set=[(X_fold_val, y_fold_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                # For linear models
                if hasattr(model_clone, 'fit'):
                    model_clone.fit(X_fold_train, y_fold_train, sample_weight=time_weights)
                else:
                    model_clone.fit(X_fold_train, y_fold_train)
            
            # Validate
            val_pred = model_clone.predict(X_fold_val)
            val_score = pearsonr(y_fold_val, val_pred)[0]
            fold_scores.append(val_score)
            
            # Store out-of-fold predictions
            oof_pred[val_idx] = val_pred
            
            # Predict on test
            test_pred = model_clone.predict(X_test_scaled)
            fold_predictions.append(test_pred)
            
            print(f"  Fold {fold + 1}: {val_score:.4f}")
        
        # Average predictions across folds
        model_predictions = np.mean(fold_predictions, axis=0)
        model_score = np.mean(fold_scores)
        
        all_predictions[model_name] = model_predictions
        model_scores[model_name] = model_score
        oof_predictions[model_name] = oof_pred
        
        print(f"  Average score: {model_score:.4f} (std: {np.std(fold_scores):.4f})")
    
    return all_predictions, model_scores, oof_predictions

def create_optimized_ensemble(predictions_dict, scores_dict, oof_predictions_dict):
    """Create optimized ensemble minimizing correlation between models"""
    print(f"\nOptimizing ensemble from {len(predictions_dict)} models...")
    
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    
    # Calculate model correlations from OOF predictions
    corr_matrix = np.zeros((n_models, n_models))
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if model1 in oof_predictions_dict and model2 in oof_predictions_dict:
                corr = pearsonr(oof_predictions_dict[model1], 
                              oof_predictions_dict[model2])[0]
                corr_matrix[i, j] = corr
    
    print("\nModel correlation matrix:")
    for i, model in enumerate(model_names):
        corr_str = " ".join([f"{corr:.2f}" for corr in corr_matrix[i]])
        print(f"  {model}: {corr_str}")
    
    if Config.use_ensemble_optimization:
        # Optimize weights
        def ensemble_objective(weights):
            weights = weights / weights.sum()
            
            # Expected score
            expected_score = sum(w * scores_dict[m] for w, m in zip(weights, model_names))
            
            # Diversity bonus (lower correlation is better)
            diversity_bonus = 0
            for i in range(n_models):
                for j in range(i+1, n_models):
                    diversity_bonus -= weights[i] * weights[j] * abs(corr_matrix[i, j])
            
            # Maximize score and diversity
            return -(expected_score + 0.3 * diversity_bonus)
        
        # Initial weights based on scores
        initial_weights = np.array([scores_dict[m] for m in model_names])
        initial_weights = initial_weights / initial_weights.sum()
        
        # Optimization bounds
        bounds = [(0.0, Config.max_model_weight) for _ in range(n_models)]
        
        # Optimize
        result = minimize(
            ensemble_objective, 
            initial_weights,
            bounds=bounds,
            method='SLSQP',
            options={'maxiter': 1000}
        )
        
        optimal_weights = result.x / result.x.sum()
    else:
        # Simple score-based weights
        scores = np.array([scores_dict[m] for m in model_names])
        optimal_weights = scores / scores.sum()
    
    print("\nOptimal ensemble weights:")
    for model, weight in zip(model_names, optimal_weights):
        print(f"  {model}: {weight:.3f}")
    
    # Create ensemble predictions
    ensemble_pred = sum(w * predictions_dict[m] 
                       for w, m in zip(optimal_weights, model_names))
    
    # Also create a simple average of top 3 models
    top_models = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    top_avg = np.mean([predictions_dict[m] for m, _ in top_models], axis=0)
    
    return ensemble_pred, top_avg, optimal_weights

def main():
    """Main execution pipeline with robust methods"""
    print("="*60)
    print("ROBUST CRYPTO PREDICTION PIPELINE")
    print("="*60)
    
    # Load data
    train, test = load_and_prepare_data()
    
    # Create comprehensive features
    print("\nEngineering features...")
    train, train_features = create_all_features(train, is_train=True)
    test, test_features = create_all_features(test, is_train=False)
    
    # Get common features
    common_features = list(set(train_features) & set(test_features))
    print(f"Common features: {len(common_features)}")
    
    # Select stable features
    selected_features = select_stable_features(train, common_features, 
                                             n_features=Config.n_top_features)
    
    # Train models with time series validation
    predictions, scores, oof_predictions = train_with_time_series_validation(
        train, test, selected_features
    )
    
    # Create optimized ensemble
    print("\n" + "="*60)
    print("CREATING ENSEMBLE PREDICTIONS")
    print("="*60)
    
    ensemble_pred, top_avg_pred, weights = create_optimized_ensemble(
        predictions, scores, oof_predictions
    )
    
    # Create final blended prediction
    final_prediction = 0.7 * ensemble_pred + 0.3 * top_avg_pred
    
    # Save submissions
    sample = pd.read_csv(Config.sample_sub_path)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    for model, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{model}: {score:.4f}")
    
    # Save best single model
    best_model = max(scores.items(), key=lambda x: x[1])[0]
    submission = sample.copy()
    submission['prediction'] = predictions[best_model]
    submission.to_csv('submission_best_single_robust.csv', index=False)
    print(f"\nSaved: submission_best_single_robust.csv ({best_model})")
    
    # Save optimized ensemble
    submission = sample.copy()
    submission['prediction'] = ensemble_pred
    submission.to_csv('submission_ensemble_robust.csv', index=False)
    print("Saved: submission_ensemble_robust.csv")
    
    # Save final blended prediction
    submission = sample.copy()
    submission['prediction'] = final_prediction
    submission.to_csv('submission_final_robust.csv', index=False)
    print("Saved: submission_final_robust.csv (RECOMMENDED)")
    
    print("\n" + "="*60)
    print("ROBUST PIPELINE COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()

