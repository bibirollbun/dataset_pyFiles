import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-darkgrid')


class CFG:
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    
    SUCCESSFUL_X_FEATURES = [
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612",
        "X888", "X421", "X333"
    ]
    
    MARKET_FEATURES = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]

def reduce_mem_usage(dataframe, dataset):    
    """Reduce memory usage by optimizing data types"""
    print(f'Reducing memory usage for: {dataset}')
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
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                dataframe[col] = dataframe[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                dataframe[col] = dataframe[col].astype(np.float32)

    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print(f'--- Memory usage before: {initial_mem_usage:.2f} MB')
    print(f'--- Memory usage after: {final_mem_usage:.2f} MB')
    print(f'--- Decreased by: {100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage:.1f}%\n')
    return dataframe

train_df = pd.read_parquet(CFG.train_path).reset_index(drop=True)
train_df = reduce_mem_usage(train_df, 'train')

test_df = pd.read_parquet(CFG.test_path).reset_index(drop=True)
test_df = reduce_mem_usage(test_df, 'test')

print(f"Data shapes - Train: {train_df.shape}, Test: {test_df.shape}")
print(f"Target stats: Mean={train_df['label'].mean():.6f}, Std={train_df['label'].std():.4f}")


def analyze_successful_features():
    """Analyze the characteristics of successful features"""
    
    print("=== ANALYSIS OF SUCCESSFUL FEATURES ===")
    
    successful_correlations = {}
    successful_stats = {}
    
    for feature in CFG.SUCCESSFUL_X_FEATURES:
        if feature in train_df.columns:
            # Correlation with target
            corr = train_df[feature].corr(train_df['label'])
            successful_correlations[feature] = corr
            
            # Statistical properties
            stats_dict = {
                'min': train_df[feature].min(),
                'max': train_df[feature].max(),
                'mean': train_df[feature].mean(),
                'std': train_df[feature].std(),
                'skew': train_df[feature].skew(),
                'unique_values': train_df[feature].nunique(),
                'correlation': corr
            }
            successful_stats[feature] = stats_dict
    
    stats_df = pd.DataFrame(successful_stats).T
    stats_df = stats_df.sort_values('correlation', key=abs, ascending=False)
    
    print("\nTop 10 successful features by correlation:")
    print(stats_df[['correlation', 'mean', 'std', 'unique_values']].head(10))
    
    plt.figure(figsize=(12, 8))
    correlations = stats_df['correlation'].values
    features = stats_df.index.tolist()
    
    colors = ['red' if abs(x) > 0.02 else 'orange' if abs(x) > 0.01 else 'blue' for x in correlations]
    plt.barh(range(len(features)), correlations, color=colors)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Correlation with Target')
    plt.title('Successful Features: Target Correlations')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.axvline(x=0.02, color='red', linestyle='--', alpha=0.5, label='|corr| > 0.02')
    plt.axvline(x=-0.02, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=0.01, color='orange', linestyle='--', alpha=0.5, label='|corr| > 0.01')
    plt.axvline(x=-0.01, color='orange', linestyle='--', alpha=0.5)
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== SUCCESSFUL FEATURES SUMMARY ===")
    print(f"Total successful features: {len(successful_correlations)}")
    print(f"Mean absolute correlation: {np.mean([abs(c) for c in successful_correlations.values()]):.4f}")
    print(f"Features with |corr| > 0.02: {sum(1 for c in successful_correlations.values() if abs(c) > 0.02)}")
    print(f"Features with |corr| > 0.01: {sum(1 for c in successful_correlations.values() if abs(c) > 0.01)}")
    
    return stats_df, successful_correlations

successful_stats_df, successful_corrs = analyze_successful_features()


def comprehensive_x_feature_analysis():
    """Analyze all X features to find more predictive ones"""
    
    print("=== COMPREHENSIVE X FEATURE ANALYSIS ===")
    
    all_x_features = [col for col in train_df.columns if col.startswith('X')]
    print(f"Total X features available: {len(all_x_features)}")
    
    all_correlations = {}
    all_stats = {}
    
    print("Calculating correlations for all X features...")
    for i, feature in enumerate(all_x_features):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(all_x_features)}")
            
        corr = train_df[feature].corr(train_df['label'])
        if not np.isnan(corr):
            all_correlations[feature] = corr
            
            if abs(corr) > 0.005:  # Only calculate detailed stats for promising features
                all_stats[feature] = {
                    'correlation': corr,
                    'mean': train_df[feature].mean(),
                    'std': train_df[feature].std(),
                    'unique_values': train_df[feature].nunique(),
                    'is_successful': feature in CFG.SUCCESSFUL_X_FEATURES
                }
    
    # Sort by absolute correlation
    sorted_correlations = sorted(all_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\nTop 50 X features by absolute correlation:")
    for i, (feature, corr) in enumerate(sorted_correlations[:50]):
        is_successful = "âœ“" if feature in CFG.SUCCESSFUL_X_FEATURES else " "
        print(f"{i+1:2d}. {feature}: {corr:7.4f} {is_successful}")
    

    strong_threshold = 0.015  # Threshold for strong correlation
    strong_features = [f for f, c in sorted_correlations if abs(c) > strong_threshold]
    new_strong_features = [f for f in strong_features if f not in CFG.SUCCESSFUL_X_FEATURES]
    
    print(f"\n=== DISCOVERY RESULTS ===")
    print(f"Features with |correlation| > {strong_threshold}: {len(strong_features)}")
    print(f"New strong features (not in successful set): {len(new_strong_features)}")
    
    if new_strong_features:
        print(f"\nNEW STRONG FEATURES TO TEST:")
        for feature in new_strong_features[:15]:  # Show top 15 new ones
            corr = all_correlations[feature]
            print(f"  {feature}: {corr:.4f}")
    
    stats_df = pd.DataFrame(all_stats).T
    
    return sorted_correlations, new_strong_features, stats_df

all_x_correlations, new_strong_features, comprehensive_stats = comprehensive_x_feature_analysis()


def advanced_feature_selection():
  """Use multiple feature selection methods - FIXED VERSION"""

  print("=== ADVANCED FEATURE SELECTION ===")

  all_x_features = [col for col in train_df.columns if col.startswith('X')]
  X = train_df[all_x_features].copy()
  y = train_df['label'].copy()

  print(f"Original shape: {X.shape}")

  X = X.replace([np.inf, -np.inf], np.nan)

  for col in X.columns:
      q99 = X[col].quantile(0.99)
      q01 = X[col].quantile(0.01)
      if not pd.isna(q99) and not pd.isna(q01):
          X[col] = X[col].clip(lower=q01, upper=q99)

  for col in X.columns:
      if X[col].isna().any():
          median_val = X[col].median()
          X[col] = X[col].fillna(median_val if not pd.isna(median_val) else 0)

  print(f"Cleaned shape: {X.shape}")

  correlation_based = [f for f, c in all_x_correlations[:100] if abs(c) > 0.005]

  print(f"Selected {len(correlation_based)} features with |correlation| > 0.005")

  consensus_features = [(f, dict(all_x_correlations)[f]) for f in correlation_based]
  consensus_features = sorted(consensus_features, key=lambda x: abs(x[1]), reverse=True)

  return consensus_features, {'correlation_based': correlation_based}

consensus_features, selection_methods = advanced_feature_selection()


def create_enhanced_feature_set():
    """Create an enhanced feature set combining successful and new features"""
    
    print("=== CREATING ENHANCED FEATURE SET ===")
    
    # Start with successful features as base
    enhanced_features = CFG.SUCCESSFUL_X_FEATURES.copy()
    print(f"Base successful features: {len(enhanced_features)}")
    
    # Add market features
    enhanced_features.extend(CFG.MARKET_FEATURES)
    print(f"After adding market features: {len(enhanced_features)}")
    
    # Add top new features from correlation analysis
    new_from_correlation = new_strong_features[:10]  # Top 10 new strong features
    enhanced_features.extend(new_from_correlation)
    print(f"After adding new correlation features: {len(enhanced_features)}")
    
    # Add top consensus features not already included
    consensus_to_add = []
    for feature, score in consensus_features[:30]:  # Check top 30 consensus
        if feature not in enhanced_features and score > 2:  # High consensus score
            consensus_to_add.append(feature)
        if len(consensus_to_add) >= 15:  # Limit to 15 additional
            break
    
    enhanced_features.extend(consensus_to_add)
    print(f"After adding consensus features: {len(enhanced_features)}")
    
    # Remove duplicates and sort
    enhanced_features = list(set(enhanced_features))
    enhanced_features.sort()
    
    print(f"\nFinal enhanced feature set: {len(enhanced_features)} features")
    
    # Calculate correlations for enhanced set
    enhanced_correlations = {}
    for feature in enhanced_features:
        if feature in train_df.columns and feature != 'label':
            corr = train_df[feature].corr(train_df['label'])
            if not np.isnan(corr):
                enhanced_correlations[feature] = corr
    
    # Sort by correlation strength
    sorted_enhanced = sorted(enhanced_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\nTop 20 features in enhanced set:")
    for i, (feature, corr) in enumerate(sorted_enhanced[:20]):
        feature_type = "Market" if feature in CFG.MARKET_FEATURES else "Successful" if feature in CFG.SUCCESSFUL_X_FEATURES else "New"
        print(f"{i+1:2d}. {feature}: {corr:7.4f} ({feature_type})")
    
    return enhanced_features, enhanced_correlations

enhanced_feature_set, enhanced_correlations = create_enhanced_feature_set()


def advanced_feature_engineering(df, feature_set):
    """Create advanced features based on successful patterns"""
    
    print("=== ADVANCED FEATURE ENGINEERING ===")
    
    result_df = df.copy()
    
    # 1. Successful microstructure features from 0.12 model
    if all(col in df.columns for col in CFG.MARKET_FEATURES):
        print("Creating microstructure features...")
        
        # Original successful features
        result_df['log_volume'] = np.log1p(df['volume'])
        result_df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-8)
        result_df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-8)
        result_df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-8)
        
        # Additional microstructure features
        result_df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
        result_df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
        result_df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-8)
        result_df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
        
        # Enhanced ratios
        result_df['volume_concentration'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + df['volume'] + 1e-8)
        result_df['aggressive_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-8)
        result_df['market_efficiency'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
        
        # Log transformations
        for col in CFG.MARKET_FEATURES:
            result_df[f'log_{col}'] = np.log1p(df[col])
    
    # 2. Top X feature interactions
    top_x_features = [f for f in enhanced_feature_set if f.startswith('X')][:10]  # Top 10 X features
    
    if len(top_x_features) >= 2:
        print(f"Creating interactions for top {len(top_x_features)} X features...")
        
        # Pairwise interactions for top features
        for i, feat1 in enumerate(top_x_features[:5]):  # Limit to avoid explosion
            if feat1 in df.columns:
                for feat2 in top_x_features[i+1:6]:  # Max 5 interactions per feature
                    if feat2 in df.columns:
                        # Multiplicative interaction
                        result_df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                        
                        # Ratio interaction
                        result_df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-8)
    
    # 3. X features with market features interactions
    print("Creating X-feature and market feature interactions...")
    for x_feat in top_x_features[:5]:  # Top 5 X features
        if x_feat in df.columns:
            for market_feat in CFG.MARKET_FEATURES:
                if market_feat in df.columns:
                    # Key interactions
                    result_df[f'{x_feat}_x_{market_feat}'] = df[x_feat] * df[market_feat]
                    result_df[f'{x_feat}_div_{market_feat}'] = df[x_feat] / (df[market_feat] + 1e-8)
    
    # 4. Polynomial features for strongest predictors
    strongest_features = [f for f, c in sorted(enhanced_correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]]
    
    print(f"Creating polynomial features for strongest predictors...")
    for feature in strongest_features:
        if feature in df.columns:
            result_df[f'{feature}_squared'] = df[feature] ** 2
            result_df[f'{feature}_sqrt'] = np.sign(df[feature]) * np.sqrt(np.abs(df[feature]))
    
    # 5. Rolling statistics (simplified for static data)
    print("Creating statistical transformations...")
    for feature in top_x_features[:3] + CFG.MARKET_FEATURES:
        if feature in df.columns:
            # Rank normalization
            result_df[f'{feature}_rank'] = df[feature].rank(pct=True)
            
            # Z-score normalization
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            if std_val > 0:
                result_df[f'{feature}_zscore'] = (df[feature] - mean_val) / std_val
    
    # Clean up inf/nan values
    result_df = result_df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with median for robustness
    for col in result_df.columns:
        if result_df[col].isna().any():
            median_val = result_df[col].median()
            result_df[col] = result_df[col].fillna(median_val if not pd.isna(median_val) else 0)
    
    new_features = [col for col in result_df.columns if col not in df.columns]
    print(f"Created {len(new_features)} new engineered features")
    
    return result_df, new_features

# Apply feature engineering
train_engineered, new_feature_names = advanced_feature_engineering(train_df, enhanced_feature_set)
test_engineered, _ = advanced_feature_engineering(test_df, enhanced_feature_set)

print(f"\nEngineered data shapes - Train: {train_engineered.shape}, Test: {test_engineered.shape}")


def optimize_feature_selection():
    """Select optimal features from enhanced set"""
    
    print("=== OPTIMIZING FEATURE SELECTION ===")
    
    # Calculate correlations for all features
    all_feature_correlations = {}
    
    for col in train_engineered.columns:
        if col != 'label':
            corr = train_engineered[col].corr(train_engineered['label'])
            if not np.isnan(corr):
                all_feature_correlations[col] = corr
    
    # Sort by absolute correlation
    sorted_all_features = sorted(all_feature_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"Total features available: {len(all_feature_correlations)}")
    
    # Feature selection criteria
    correlation_threshold = 0.008  # Minimum correlation threshold
    max_features = 60  # Maximum number of features
    
    # Select features above threshold
    selected_features = []
    selected_correlations = {}
    
    for feature, corr in sorted_all_features:
        if abs(corr) >= correlation_threshold and len(selected_features) < max_features:
            selected_features.append(feature)
            selected_correlations[feature] = corr
    
    print(f"Selected {len(selected_features)} features with |correlation| >= {correlation_threshold}")
    
    # Analyze feature types
    feature_types = {
        'original_x': [f for f in selected_features if f.startswith('X') and f in CFG.SUCCESSFUL_X_FEATURES],
        'new_x': [f for f in selected_features if f.startswith('X') and f not in CFG.SUCCESSFUL_X_FEATURES],
        'market': [f for f in selected_features if f in CFG.MARKET_FEATURES],
        'engineered': [f for f in selected_features if f in new_feature_names]
    }
    
    print(f"\n=== FEATURE BREAKDOWN ===")
    for ftype, features in feature_types.items():
        print(f"{ftype}: {len(features)} features")
        if features:
            avg_corr = np.mean([abs(selected_correlations[f]) for f in features])
            print(f"  Average |correlation|: {avg_corr:.4f}")
    
    print(f"\nTop 20 selected features:")
    for i, feature in enumerate(selected_features[:20]):
        corr = selected_correlations[feature]
        ftype = 'Market' if feature in CFG.MARKET_FEATURES else 'Orig-X' if feature in CFG.SUCCESSFUL_X_FEATURES else 'New-X' if feature.startswith('X') else 'Eng'
        print(f"{i+1:2d}. {feature}: {corr:7.4f} ({ftype})")
    
    return selected_features, selected_correlations, feature_types

optimal_features, optimal_correlations, feature_breakdown = optimize_feature_selection()


def quick_model_validation():
    """Quick validation of enhanced feature set"""
    
    print("=== QUICK MODEL VALIDATION ===")
    
    # Prepare data
    X_train = train_engineered[optimal_features].fillna(0)
    y_train = train_engineered['label']
    X_test = test_engineered[optimal_features].fillna(0)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Simple time series split for validation
    split_idx = int(0.8 * len(X_train))
    
    X_train_split = X_train.iloc[:split_idx]
    X_val_split = X_train.iloc[split_idx:]
    y_train_split = y_train.iloc[:split_idx]
    y_val_split = y_train.iloc[split_idx:]
    
    print(f"Training split: {X_train_split.shape}")
    print(f"Validation split: {X_val_split.shape}")
    
    results = {}
    
    # Model 1: LightGBM
    print("\n1. Testing LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=10,
        reg_lambda=10,
        random_state=42,
        verbose=-1
    )
    
    lgb_model.fit(X_train_split, y_train_split, 
                  eval_set=[(X_val_split, y_val_split)],
                  callbacks=[])
    
    lgb_train_pred = lgb_model.predict(X_train)
    lgb_val_pred = lgb_model.predict(X_val_split)
    lgb_test_pred = lgb_model.predict(X_test)
    
    lgb_train_corr = pearsonr(y_train, lgb_train_pred)[0]
    lgb_val_corr = pearsonr(y_val_split, lgb_val_pred)[0]
    
    results['LightGBM'] = {
        'train_corr': lgb_train_corr,
        'val_corr': lgb_val_corr,
        'test_pred': lgb_test_pred
    }
    
    print(f"LightGBM - Train: {lgb_train_corr:.4f}, Val: {lgb_val_corr:.4f}")
    
    # Model 2: XGBoost (simplified)
    print("\n2. Testing XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=10,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=20,
        reg_lambda=40,
        random_state=42,
        verbosity=0
    )
    
    xgb_model.fit(X_train_split, y_train_split,
                  eval_set=[(X_val_split, y_val_split)],
                  verbose=False)
    
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_val_pred = xgb_model.predict(X_val_split)
    xgb_test_pred = xgb_model.predict(X_test)
    
    xgb_train_corr = pearsonr(y_train, xgb_train_pred)[0]
    xgb_val_corr = pearsonr(y_val_split, xgb_val_pred)[0]
    
    results['XGBoost'] = {
        'train_corr': xgb_train_corr,
        'val_corr': xgb_val_corr,
        'test_pred': xgb_test_pred
    }
    
    print(f"XGBoost - Train: {xgb_train_corr:.4f}, Val: {xgb_val_corr:.4f}")
    
    # Simple ensemble
    ensemble_test_pred = 0.5 * lgb_test_pred + 0.5 * xgb_test_pred
    ensemble_train_pred = 0.5 * lgb_train_pred + 0.5 * xgb_train_pred
    ensemble_train_corr = pearsonr(y_train, ensemble_train_pred)[0]
    
    results['Ensemble'] = {
        'train_corr': ensemble_train_corr,
        'val_corr': (lgb_val_corr + xgb_val_corr) / 2,  # Approximate
        'test_pred': ensemble_test_pred
    }
    
    print(f"Ensemble - Train: {ensemble_train_corr:.4f}")
    
    # Feature importance analysis
    print(f"\n=== FEATURE IMPORTANCE (LightGBM) ===")
    feature_importance = pd.DataFrame({
        'feature': optimal_features,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 15 most important features:")
    for i, row in feature_importance.head(15).iterrows():
        ftype = 'Market' if row['feature'] in CFG.MARKET_FEATURES else 'Orig-X' if row['feature'] in CFG.SUCCESSFUL_X_FEATURES else 'New-X' if row['feature'].startswith('X') else 'Eng'
        print(f"{row.name+1:2d}. {row['feature']}: {row['importance']:8.1f} ({ftype})")
    
    return results, feature_importance

validation_results, feature_importance_df = quick_model_validation()


def generate_enhanced_submissions():
    """Generate submission files with enhanced features"""
    
    print("=== GENERATING ENHANCED SUBMISSIONS ===")
    
    # Load sample submission format
    sample_submission = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")
    
    # Create submissions for each model
    submissions = {}
    
    for model_name, results in validation_results.items():
        submission = sample_submission.copy()
        submission['prediction'] = results['test_pred']
        
        filename = f"enhanced_{model_name.lower()}_submission.csv"
        submission.to_csv(filename, index=False)
        
        submissions[model_name] = {
            'filename': filename,
            'train_corr': results['train_corr'],
            'val_corr': results['val_corr'],
            'pred_stats': {
                'min': results['test_pred'].min(),
                'max': results['test_pred'].max(),
                'mean': results['test_pred'].mean(),
                'std': results['test_pred'].std()
            }
        }
        
        print(f"âœ… Saved {filename}")
        print(f"   Train correlation: {results['train_corr']:.4f}")
        print(f"   Validation correlation: {results['val_corr']:.4f}")
        print(f"   Prediction range: [{results['test_pred'].min():.3f}, {results['test_pred'].max():.3f}]")
        print()
    
    return submissions

submission_summary = generate_enhanced_submissions()


def final_summary_and_recommendations():
    """Provide final summary and next steps"""
    
    print("=" * 80)
    print("ENHANCED FEATURE DISCOVERY - FINAL SUMMARY")
    print("=" * 80)
    
    print(f"\nğŸ�¯ FEATURE DISCOVERY RESULTS:")
    print(f"â€¢ Analyzed {len([col for col in train_df.columns if col.startswith('X')])} total X features")
    print(f"â€¢ Found {len(new_strong_features)} new strong features (|corr| > 0.015)")
    print(f"â€¢ Created {len(new_feature_names)} engineered features")
    print(f"â€¢ Final optimal set: {len(optimal_features)} features")
    
    print(f"\nğŸ“Š MODEL PERFORMANCE:")
    for model_name, results in validation_results.items():
        print(f"â€¢ {model_name}: Train {results['train_corr']:.4f}, Val {results['val_corr']:.4f}")
    
    print(f"\nğŸ�† BEST PERFORMERS:")
    best_val = max(validation_results.items(), key=lambda x: abs(x[1]['val_corr']))
    print(f"â€¢ Best validation: {best_val[0]} ({best_val[1]['val_corr']:.4f})")
    
    print(f"\nğŸ”§ KEY IMPROVEMENTS IDENTIFIED:")
    print(f"â€¢ New predictive X features: {', '.join(new_strong_features[:5])}")
    print(f"â€¢ Most important engineered features from top 5:")
    
    top_engineered = feature_importance_df[feature_importance_df['feature'].isin(new_feature_names)].head(5)
    for _, row in top_engineered.iterrows():
        print(f"  - {row['feature']} (importance: {row['importance']:.1f})")
    
    
    print(f"\nğŸ“� FILES CREATED:")
    for model_name, details in submission_summary.items():
        print(f"â€¢ {details['filename']} - {model_name} model")
    
    
    print("\n" + "=" * 80)

final_summary_and_recommendations()

