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


# ============================
# LightGBM V10_lean Model - Production Version
# - Best parameters from Optuna optimization
# - Train with best params on 20 folds
# - OOF and test predictions saved as .npy files
# - Feature importance and comprehensive residual analysis
# ============================

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
import random
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.simplefilter('ignore')
warnings.filterwarnings("ignore")

seed = 42
random.seed(seed)
np.random.seed(seed)

# Date stamp for file naming
date_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = "V10_lean"

# -----------------------------
# Best Parameters from Optuna
# -----------------------------
BEST_PARAMS = {
    'objective': 'mae',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': seed,
    'force_col_wise': True,
    'num_threads': -1,
    'num_leaves': 172,
    'max_depth': 10,
    'min_data_in_leaf': 27,
    'learning_rate': 0.07912129649576233,
    'feature_fraction': 0.5474489466749426,
    'bagging_fraction': 0.8903520404557033,
    'bagging_freq': 3,
    'min_gain_to_split': 0.1238372018474428,
    'cat_smooth': 3.261058476952931,
    'cat_l2': 2.795255589544235,
    'max_cat_threshold': 21,
    'lambda_l1': 0.07531704429471647,
    'lambda_l2': 0.044230712158467765,
}

# -----------------------------
# Utilities
# -----------------------------
def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize memory usage of dataframe."""
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype.name
        if col_type not in ['object', 'category', 'datetime64[ns, UTC]']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    if start_mem > 0:
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def create_price_bins_for_stratification(y, n_bins=20):
    """Create price bins for stratified K-fold."""
    y_original = 10 ** y
    return pd.qcut(y_original, q=n_bins, duplicates='drop', labels=False)


def encode_categorical_features(X_train, X_val, X_test, cat_features):
    """Encode categorical features with proper handling of unseen values."""
    encoders = {}
    for col in cat_features:
        if col in X_train.columns:
            enc = LabelEncoder()
            tr = X_train[col].astype(str)
            enc.fit(tr)
            X_train[col] = enc.transform(tr)
            for ds in (X_val, X_test):
                s = ds[col].astype(str)
                known = set(enc.classes_)
                unseen = ~s.isin(known)
                if unseen.any():
                    if 'UNK' not in known:
                        enc.classes_ = np.append(enc.classes_, 'UNK')
                    s2 = s.copy()
                    s2[unseen] = 'UNK'
                    ds[col] = enc.transform(s2)
                else:
                    ds[col] = enc.transform(s)
            encoders[col] = enc
    return X_train, X_val, X_test, encoders


# -----------------------------
# V10_lean Feature Recipe
# -----------------------------
def create_v10_lean_features(df):
    """V10: Lean engineered set with few strong signals."""
    df = df.copy()
    
    # Cyclical encoding of month
    df['month_sin'] = np.sin(2 * np.pi * df['sale_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['sale_month'] / 12)
    
    # Distance from London center
    london_center_lat, london_center_lon = 51.5074, -0.1278
    df['dist_from_center'] = np.sqrt(
        (df['latitude'] - london_center_lat)**2 + 
        (df['longitude'] - london_center_lon)**2
    )
    
    # Room-based features
    df['total_rooms'] = df['bedrooms'] + df['livingRooms'] + df['bathrooms']
    df['room_density'] = df['total_rooms'] / (df['floorAreaSqM'] + 1)
    df['area_per_bedroom'] = df['floorAreaSqM'] / (df['bedrooms'] + 1)
    
    # Energy rating numeric
    energy_mapping = {
        'A': 10, 'B': 9, 'C': 8, 'D': 7, 'E': 6, 'F': 5, 'G': 4,
        'A+': 11, 'A++': 12
    }
    df['energy_rating_numeric'] = df['currentEnergyRating'].map(energy_mapping).fillna(7)
    
    # Premium score
    df['premium_score'] = (
        (df['floorAreaSqM'] / 100) * 
        (1 / (df['dist_from_center'] + 0.1)) * 
        (df['energy_rating_numeric'] / 10)
    )
    
    return df


# -----------------------------
# Residual Analysis Function
# -----------------------------
def analyze_residuals(y_true, y_pred, fold_id, save_prefix):
    """Comprehensive residual analysis with visualizations."""
    
    # Convert back to original scale
    y_true_orig = 10 ** y_true
    y_pred_orig = 10 ** y_pred
    
    # Calculate residuals
    residuals_log = y_true - y_pred
    residuals_orig = y_true_orig - y_pred_orig
    residuals_pct = (residuals_orig / y_true_orig) * 100
    
    # Calculate metrics
    mae_log = mean_absolute_error(y_true, y_pred)
    mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2 = r2_score(y_true, y_pred)
    
    # Statistical tests
    _, normality_p = stats.normaltest(residuals_log)
    
    # Percentile analysis
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(np.abs(residuals_orig), percentiles)
    
    # Create comprehensive analysis dictionary
    analysis = {
        'fold': fold_id,
        'n_samples': len(y_true),
        'mae_log10': mae_log,
        'mae_original': mae_orig,
        'rmse_log10': rmse_log,
        'rmse_original': rmse_orig,
        'r2_score': r2,
        'mean_residual_log': np.mean(residuals_log),
        'std_residual_log': np.std(residuals_log),
        'mean_residual_orig': np.mean(residuals_orig),
        'std_residual_orig': np.std(residuals_orig),
        'mean_abs_pct_error': np.mean(np.abs(residuals_pct)),
        'median_abs_pct_error': np.median(np.abs(residuals_pct)),
        'max_overestimation': np.min(residuals_orig),
        'max_underestimation': np.max(residuals_orig),
        'normality_p_value': normality_p,
        'skewness': stats.skew(residuals_log),
        'kurtosis': stats.kurtosis(residuals_log),
    }
    
    # Add percentiles
    for p, v in zip(percentiles, percentile_values):
        analysis[f'abs_error_{p}th_percentile'] = v
    
    return analysis


def print_residual_summary(residual_analyses):
    """Print detailed residual analysis summary to console."""
    print("\n" + "="*80)
    print("DETAILED RESIDUAL ANALYSIS SUMMARY")
    print("="*80)
    
    # Create DataFrame for easy manipulation
    df = pd.DataFrame(residual_analyses)
    
    # Overall statistics
    print("\nOVERALL STATISTICS (averaged across all folds):")
    print("-" * 80)
    
    metrics = [
        ('MAE (log10)', 'mae_log10'),
        ('MAE (£)', 'mae_original'),
        ('RMSE (log10)', 'rmse_log10'),
        ('RMSE (£)', 'rmse_original'),
        ('R² Score', 'r2_score'),
        ('Mean Abs % Error', 'mean_abs_pct_error'),
        ('Median Abs % Error', 'median_abs_pct_error'),
    ]
    
    for label, col in metrics:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if '£' in label:
            print(f"  {label:25s}: £{mean_val:>12,.2f} (±{std_val:,.2f})")
        elif '%' in label:
            print(f"  {label:25s}: {mean_val:>12.2f}% (±{std_val:.2f}%)")
        else:
            print(f"  {label:25s}: {mean_val:>12.6f} (±{std_val:.6f})")
    
    # Error percentiles
    print("\nERROR DISTRIBUTION (averaged across all folds):")
    print("-" * 80)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        col_name = f'abs_error_{p}th_percentile'
        mean_val = df[col_name].mean()
        print(f"  {p:2d}th percentile: £{mean_val:>12,.2f}")
    
    # Residual statistics
    print("\nRESIDUAL STATISTICS (log scale):")
    print("-" * 80)
    print(f"  Mean residual:        {df['mean_residual_log'].mean():>12.6f}")
    print(f"  Std deviation:        {df['std_residual_log'].mean():>12.6f}")
    print(f"  Skewness:             {df['skewness'].mean():>12.6f}")
    print(f"  Kurtosis:             {df['kurtosis'].mean():>12.6f}")
    print(f"  Normality p-value:    {df['normality_p_value'].mean():>12.6f}")
    
    # Extreme errors
    print("\nEXTREME ERRORS (averaged across all folds):")
    print("-" * 80)
    print(f"  Max overestimation:   £{df['max_overestimation'].mean():>12,.2f}")
    print(f"  Max underestimation:  £{df['max_underestimation'].mean():>12,.2f}")
    
    # Fold-by-fold comparison
    print("\nFOLD-BY-FOLD COMPARISON:")
    print("-" * 80)
    print(f"{'Fold':<6} {'MAE (£)':>15} {'RMSE (£)':>15} {'R²':>10} {'Median % Err':>15}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"{int(row['fold']):<6} "
              f"£{row['mae_original']:>13,.2f} "
              f"£{row['rmse_original']:>13,.2f} "
              f"{row['r2_score']:>10.4f} "
              f"{row['median_abs_pct_error']:>14.2f}%")
    
    print("="*80)


# -----------------------------
# Main Training Function
# -----------------------------
def main():
    print("="*80)
    print(f"V10_lean Model Training - Production Version")
    print(f"Date: {date_stamp}")
    print("="*80)
    
    print("\nUsing optimized parameters:")
    for key, value in BEST_PARAMS.items():
        if key not in ['objective', 'metric', 'boosting_type', 'verbosity', 'random_state', 'force_col_wise', 'num_threads']:
            print(f"  {key}: {value}")
    
    # Load data
    print("\nLoading data...")
    data = pd.read_csv('/kaggle/input/london-house-price-prediction-advanced-techniques/train.csv')
    test = pd.read_csv('/kaggle/input/london-house-price-prediction-advanced-techniques/test.csv')
    print(f"Data shapes - Train: {data.shape}, Test: {test.shape}")
    
    # Remove columns with >50% missing values
    dt = []
    for col in test.columns:
        if test[col].isnull().sum() > len(test) * 0.5:
            dt.append(col)
    for col in data.columns:
        if data[col].isnull().sum() > len(data) * 0.5:
            dt.append(col)
    dt = list(set(dt))
    if dt:
        print(f"Dropping columns with >50% missing values: {dt}")
        data = data.drop(columns=dt)
        test = test.drop(columns=dt)
    
    # Memory optimization
    data = reduce_mem_usage(data)
    test = reduce_mem_usage(test)
    
    # Prepare target and features
    X_full = data.drop(columns=['ID', 'price'])
    y_full = np.log10(data['price'])
    X_test_full = test.drop(columns=['ID'])
    test_ids = test['ID'].values
    
    # Define core features
    core_features = [
        'postcode', 'country', 'outcode', 'latitude', 'longitude',
        'bathrooms', 'bedrooms', 'floorAreaSqM', 'livingRooms',
        'tenure', 'propertyType', 'currentEnergyRating', 'sale_month', 'sale_year'
    ]
    base_cols_available = [c for c in core_features if c in X_full.columns]
    
    # Keep only base columns
    X_full = X_full[base_cols_available]
    X_test_full = X_test_full[base_cols_available]
    
    # Create stratified folds
    print("\nCreating 20 stratified folds...")
    bins = create_price_bins_for_stratification(y_full, n_bins=20)
    skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=seed)
    fold_splits = list(skf.split(X_full, bins))
    
    # Training on 20 folds
    print("\n" + "="*80)
    print("Training on 20 Folds")
    print("="*80)
    
    oof_predictions = np.zeros(len(X_full))
    test_predictions = np.zeros(len(X_test_full))
    fold_scores = []
    residual_analyses = []
    feature_importance_list = []
    
    for fold_num, (train_idx, val_idx) in enumerate(fold_splits, 1):
        print(f"\n{'='*80}")
        print(f"Training Fold {fold_num}/20")
        print(f"{'='*80}")
        
        X_tr = create_v10_lean_features(X_full.iloc[train_idx].copy())
        X_va = create_v10_lean_features(X_full.iloc[val_idx].copy())
        X_te = create_v10_lean_features(X_test_full.copy())
        y_tr = y_full.iloc[train_idx]
        y_va = y_full.iloc[val_idx]
        
        # Get feature columns
        feature_cols = X_tr.columns.tolist()
        cat_features = [c for c in feature_cols if c in [
            'postcode', 'country', 'outcode', 'tenure', 'propertyType', 'currentEnergyRating'
        ]]
        
        # Encode categorical features
        X_tr_enc, X_va_enc, X_te_enc, _ = encode_categorical_features(
            X_tr.copy(), X_va.copy(), X_te.copy(), cat_features
        )
        
        # Create datasets
        cat_inds = [i for i, col in enumerate(feature_cols) if col in cat_features]
        dtr = lgb.Dataset(X_tr_enc[feature_cols], label=y_tr, categorical_feature=cat_inds)
        dva = lgb.Dataset(X_va_enc[feature_cols], label=y_va, categorical_feature=cat_inds, reference=dtr)
        
        # Train model
        model = lgb.train(
            BEST_PARAMS,
            dtr,
            valid_sets=[dva],
            num_boost_round=18000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
        )
        
        # Make predictions
        val_pred = model.predict(X_va_enc[feature_cols])
        test_pred = model.predict(X_te_enc[feature_cols])
        
        # Store OOF predictions
        oof_predictions[val_idx] = val_pred
        test_predictions += test_pred / 20
        
        # Calculate metrics
        mae_log = mean_absolute_error(y_va, val_pred)
        mae_orig = mean_absolute_error(10**y_va, 10**val_pred)
        rmse_log = np.sqrt(mean_squared_error(y_va, val_pred))
        rmse_orig = np.sqrt(mean_squared_error(10**y_va, 10**val_pred))
        
        fold_scores.append({
            'fold': fold_num,
            'mae_log10': mae_log,
            'mae_original': mae_orig,
            'rmse_log10': rmse_log,
            'rmse_original': rmse_orig,
            'best_iteration': model.best_iteration
        })
        
        print(f"Fold {fold_num} - MAE(log10): {mae_log:.6f} | MAE(£): £{mae_orig:,.0f} | RMSE(log10): {rmse_log:.6f}")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importance(importance_type='gain'),
            'fold': fold_num
        })
        feature_importance_list.append(importance_df)
        
        # Residual analysis
        residual_analysis = analyze_residuals(y_va, val_pred, fold_num, f"{date_stamp}_{model_name}")
        residual_analyses.append(residual_analysis)
    
    # Calculate overall OOF scores
    oof_mae_log = mean_absolute_error(y_full, oof_predictions)
    oof_mae_orig = mean_absolute_error(10**y_full, 10**oof_predictions)
    oof_rmse_log = np.sqrt(mean_squared_error(y_full, oof_predictions))
    oof_rmse_orig = np.sqrt(mean_squared_error(10**y_full, 10**oof_predictions))
    
    print("\n" + "="*80)
    print("20-FOLD CROSS-VALIDATION RESULTS")
    print("="*80)
    print(f"Overall OOF MAE (log10): {oof_mae_log:.6f}")
    print(f"Overall OOF MAE (£): £{oof_mae_orig:,.0f}")
    print(f"Overall OOF RMSE (log10): {oof_rmse_log:.6f}")
    print(f"Overall OOF RMSE (£): £{oof_rmse_orig:,.0f}")
    
    # Print detailed residual analysis to console
    print_residual_summary(residual_analyses)
    
    # Save fold scores
    fold_scores_df = pd.DataFrame(fold_scores)
    print("\nFold-wise scores:")
    print(fold_scores_df.to_string(index=False))
    
    cv_filename = f"{date_stamp}_{model_name}_MAE_{int(oof_mae_orig)}_cv_scores.csv"
    fold_scores_df.to_csv(cv_filename, index=False)
    print(f"\nSaved CV scores: {cv_filename}")
    
    # Save residual analysis summary
    residual_df = pd.DataFrame(residual_analyses)
    residual_filename = f"{date_stamp}_{model_name}_MAE_{int(oof_mae_orig)}_residual_analysis.csv"
    residual_df.to_csv(residual_filename, index=False)
    print(f"Saved residual analysis: {residual_filename}")
    
    # Aggregate and save feature importance
    feature_importance_df = pd.concat(feature_importance_list, ignore_index=True)
    avg_importance = feature_importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    
    print("\n" + "="*80)
    print("TOP 20 MOST IMPORTANT FEATURES (averaged across 20 folds)")
    print("="*80)
    print(avg_importance.head(20).to_string())
    
    # Save COMPLETE feature importance (all features)
    importance_filename = f"{date_stamp}_{model_name}_MAE_{int(oof_mae_orig)}_feature_importance.csv"
    avg_importance_df = avg_importance.reset_index()
    avg_importance_df.columns = ['feature', 'importance']
    avg_importance_df.to_csv(importance_filename, index=False)
    print(f"\nSaved complete feature importance ({len(avg_importance_df)} features): {importance_filename}")
    
    # Create feature importance plot
    plt.figure(figsize=(12, 8))
    top_features = avg_importance.head(30)
    plt.barh(range(len(top_features)), top_features.values)
    plt.yticks(range(len(top_features)), top_features.index)
    plt.xlabel('Importance (Gain)', fontsize=12)
    plt.title(f'Top 30 Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    importance_plot_filename = f"{date_stamp}_{model_name}_MAE_{int(oof_mae_orig)}_feature_importance.png"
    plt.savefig(importance_plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance plot: {importance_plot_filename}")
    
    # Save OOF predictions with model name and MAE in filename
    oof_filename = f"{date_stamp}_{model_name}_MAE_{int(oof_mae_orig)}_oof_predictions.npy"
    np.save(oof_filename, oof_predictions)
    print(f"\nSaved OOF predictions: {oof_filename}")
    
    # Save test predictions with model name and MAE in filename
    test_filename = f"{date_stamp}_{model_name}_MAE_{int(oof_mae_orig)}_test_predictions.npy"
    np.save(test_filename, test_predictions)
    print(f"Saved test predictions: {test_filename}")
    
    # Create submission file
    submission = pd.DataFrame({
        'ID': test_ids,
        'price': 10 ** test_predictions
    })
    submission_filename = "submission.csv"
    submission.to_csv(submission_filename, index=False)
    print(f"Saved submission file: {submission_filename}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Final OOF MAE: £{oof_mae_orig:,.0f}")
    print(f"Final OOF RMSE: £{oof_rmse_orig:,.0f}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"All outputs saved with timestamp: {date_stamp}")
    

if __name__ == "__main__":
    main()

