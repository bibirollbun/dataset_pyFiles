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


import numpy as np
import pandas as pd
import scipy.stats
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Helper function for RMSE
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ===========================
# CONFIGURATION
# ===========================
class Config:
    target = 'accident_risk'
    state = 42
    n_splits = 7
    early_stop = 100
    metric = 'rmse'
    
print("="*80)
print("CATBOOST ROAD ACCIDENT RISK PREDICTION")
print("OPTIMIZED HYPERPARAMETERS | CURVATURE TERCILES | ERROR-GROUP FEATURES")
print("="*80)

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col='id')

# Load ALL original data files
orig = pd.concat(
    [pd.read_csv(f"/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_{k}k.csv") 
     for k in (2, 10, 100)],
    axis=0,
    ignore_index=True
)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Original data shape: {orig.shape}")

# ===========================
# FEATURE ENGINEERING WITH CURVATURE TERCILES
# ===========================
print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

def feature_engineering_curvature_terciles(data, orig, target='accident_risk'):
    """
    Feature engineering with terciles for curvature, quartiles for others.
    
    Key Features:
    - Target encoding from original data (mean/median)
    - Terciles (3 bins) for curvature
    - Quartiles (4 bins) for other numerical features
    - Squared features for non-linear relationships
    - Bayesian baseline for residual learning
    """
    mean = orig[target].mean()
    median = orig[target].median()
    
    num_features = data.select_dtypes(exclude=['object', 'bool']).columns.tolist()
    cat_features = data.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    if target in num_features:
        num_features.remove(target)
    if target in cat_features:
        cat_features.remove(target)
    
    # Target encoding from original data
    for c in num_features + cat_features:
        tmp = (orig.groupby(c)[target]           
            .agg(['mean', 'median'])
            .rename(columns=lambda a: f'{c}_org_{a}')
            .reset_index())
        data = data.merge(tmp, on=c, how='left')
    
    data['curvature_org_mean'] = data['curvature_org_mean'].fillna(mean)
    data['curvature_org_median'] = data['curvature_org_median'].fillna(median)
    
    # TERCILES for curvature (3 bins), QUARTILES for others (4 bins)
    for c in num_features:
        if c == 'curvature':
            data[f"{c}_tercile"] = pd.cut(data[c], bins=3, labels=False, include_lowest=True).astype('category')
        else:
            data[f"{c}_quartile"] = pd.cut(data[c], bins=4, labels=False, include_lowest=True).astype('category')
    
    # Squared features
    for c in ['curvature', 'speed_limit']:
        data[f"{c}_sq"] = data[c]**2
    
    # High risk interaction
    data["is_high_speed_night"] = ((data["speed_limit"] > 60) & (data["lighting"] == "night")).astype(int)
    
    # Bayesian baseline
    def f(X):
        return \
        0.3 * X["curvature"] + \
        0.2 * (X["lighting"] == "night").astype(int) + \
        0.1 * (X["weather"] != "clear").astype(int) + \
        0.2 * (X["speed_limit"] >= 60).astype(int) + \
        0.1 * (X["num_reported_accidents"] > 2).astype(int)
    
    def clip(f):
        def clip_f(X):
            sigma = 0.05
            mu = f(X)
            a, b = -mu/sigma, (1-mu)/sigma
            Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
            phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
            return mu*(Phi_b-Phi_a) + sigma*(phi_a-phi_b) + 1 - Phi_b
        return clip_f
    
    data["bayes_baseline"] = clip(f)(data).values
    
    cat_features_final = data.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
    for c in cat_features_final:
        data[c] = data[c].astype('category')
    
    return data, cat_features_final

# Apply feature engineering
train_processed, cat_features = feature_engineering_curvature_terciles(train.copy(), orig, Config.target)
test_processed, _ = feature_engineering_curvature_terciles(test.copy(), orig, Config.target)

# Prepare target
y = train_processed[Config.target].values
X = train_processed.drop(Config.target, axis=1)
X_test = test_processed

print(f"\nFinal feature count: {X.shape[1]}")
print(f"Categorical features: {len(cat_features)}")
print(f"Numerical features: {X.shape[1] - len(cat_features)}")
print(f"\n✓ Curvature uses TERCILES (3 bins)")
print(f"✓ Other numerical features use QUARTILES (4 bins)")

# ===========================
# HELPER FUNCTION FOR ERROR GROUPS
# ===========================
def create_error_group_features(train_data, baseline_preds_val, val_idx, high_error_threshold, target_col):
    """
    Create error group features based on baseline predictions.
    
    This identifies samples with high prediction errors and creates
    separate target encodings for high-error vs low-error groups.
    """
    train_copy = train_data.copy()
    
    train_copy['error_group'] = 0
    baseline_residuals = np.abs(train_copy.iloc[val_idx][target_col] - baseline_preds_val)
    high_error_indices = train_copy.iloc[val_idx][baseline_residuals > high_error_threshold].index
    train_copy.loc[high_error_indices, 'error_group'] = 1
    
    global_mean = train_copy[target_col].mean()
    
    group_cols = []
    for c in ['curvature', 'speed_limit', 'lighting', 'weather', 'road_type', 'time_of_day']:
        if train_copy[c].dtype.name == 'category':
            c_values = train_copy[c].astype(str)
        else:
            c_values = train_copy[c]
            
        for group in [0, 1]:
            mask = train_copy['error_group'] == group
            if mask.sum() > 0:
                if train_copy[c].dtype.name == 'category':
                    tmp = train_copy[mask].groupby(train_copy[mask][c].astype(str))[target_col].mean()
                    col_name = f'TE_{c}_group{group}'
                    train_copy[col_name] = c_values.map(tmp).fillna(global_mean).astype('float32')
                else:
                    tmp = train_copy[mask].groupby(c)[target_col].mean()
                    col_name = f'TE_{c}_group{group}'
                    train_copy[col_name] = train_copy[c].map(tmp).fillna(global_mean).astype('float32')
                group_cols.append(col_name)
    
    return train_copy, group_cols, global_mean

# ===========================
# INITIALIZE CROSS-VALIDATION
# ===========================
kf = KFold(n_splits=Config.n_splits, shuffle=True, random_state=Config.state)

# ===========================
# OPTIMIZED CATBOOST PARAMETERS
# ===========================
print("\n" + "="*80)
print("MODEL CONFIGURATION")
print("="*80)

catboost_params = {
    'learning_rate': 0.018122532129277353,
    'depth': 10,
    'min_data_in_leaf': 15,
    'l2_leaf_reg': 6.529176127473378,
    'bagging_temperature': 0.7521544976736819,
    'random_strength': 1.1461173031767495,
    'border_count': 224,
    'grow_policy': 'Lossguide',
    'verbose': 0,
    'random_state': Config.state,
    'cat_features': cat_features,
    'early_stopping_rounds': Config.early_stop,
    'eval_metric': "RMSE",
    'iterations': 5000,
    'task_type': "GPU"
}

print("\nOptimized CatBoost Parameters:")
print(f"  Learning rate:        {catboost_params['learning_rate']:.6f}")
print(f"  Depth:                {catboost_params['depth']}")
print(f"  Min data in leaf:     {catboost_params['min_data_in_leaf']}")
print(f"  L2 regularization:    {catboost_params['l2_leaf_reg']:.4f}")
print(f"  Bagging temperature:  {catboost_params['bagging_temperature']:.4f}")
print(f"  Random strength:      {catboost_params['random_strength']:.4f}")
print(f"  Border count:         {catboost_params['border_count']}")
print(f"  Grow policy:          {catboost_params['grow_policy']}")
print(f"  Task type:            {catboost_params['task_type']}")

# ===========================
# CROSS-VALIDATION TRAINING
# ===========================
print("\n" + "="*80)
print("TRAINING WITH 7-FOLD CROSS-VALIDATION")
print("="*80)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_scores = []
feature_importance_fold1 = None

# Storage for visualization data
visualization_data = {
    'residuals_before_log': [],
    'residuals_after_log': [],
    'errors_distribution': [],
    'error_groups': []
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n{'='*35}")
    print(f"Fold {fold+1}/{Config.n_splits}")
    print(f"{'='*35}")
    
    # Split data
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    
    # Get baseline values for residual learning
    y_train_baseline = X_train['bayes_baseline'].values
    y_val_baseline = X_val['bayes_baseline'].values
    y_test_baseline = X_test['bayes_baseline'].values
    
    # Step 1: Train baseline model to identify error groups
    print("  Step 1: Training baseline model...")
    X_train_base = X_train.copy()
    X_val_base = X_val.copy()
    
    # Train on residuals
    y_train_residual = y_train - y_train_baseline
    
    model_baseline = CatBoostRegressor(**catboost_params)
    model_baseline.fit(
        X_train_base, y_train_residual,
        eval_set=(X_val_base, y_val - y_val_baseline),
        verbose=False
    )
    
    # Get baseline predictions
    preds_baseline = model_baseline.predict(X_val_base) + y_val_baseline
    
    # Identify high error samples
    abs_errors = np.abs(y_val - preds_baseline)
    high_error_threshold = np.percentile(abs_errors, 75)
    
    # Store for visualization (fold 1 only)
    if fold == 0:
        visualization_data['errors_distribution'] = abs_errors
        visualization_data['error_groups'] = (abs_errors > high_error_threshold).astype(int)
    
    # Step 2: Create error group features
    print("  Step 2: Creating error group features...")
    
    # Reconstruct full train dataframe with target for error group feature creation
    train_full = X.copy()
    train_full[Config.target] = y
    
    train_with_groups, group_cols, global_mean = create_error_group_features(
        train_full, preds_baseline, val_idx, high_error_threshold, Config.target
    )
    
    # Apply same error group encodings to test set
    test_with_groups = X_test.copy()
    for col in group_cols:
        if col not in test_with_groups.columns:
            parts = col.replace('TE_', '').rsplit('_group', 1)
            orig_feat = parts[0]
            group = int(parts[1])
            
            mask = train_with_groups['error_group'] == group
            if mask.sum() > 0:
                # Handle categorical columns
                if test_with_groups[orig_feat].dtype.name == 'category':
                    tmp = train_with_groups[mask].groupby(train_with_groups[mask][orig_feat].astype(str))[Config.target].mean()
                    test_with_groups[col] = test_with_groups[orig_feat].astype(str).map(tmp).fillna(global_mean).astype('float32')
                else:
                    tmp = train_with_groups[mask].groupby(orig_feat)[Config.target].mean()
                    test_with_groups[col] = test_with_groups[orig_feat].map(tmp).fillna(global_mean).astype('float32')
    
    print(f"  → Added {len(group_cols)} error-group features")
    
    # Step 3: Train final model with variance stabilization
    print("  Step 3: Training final model...")
    
    # Prepare data with error group features
    X_train_combined = train_with_groups.iloc[train_idx].drop(columns=[Config.target, 'error_group'])
    y_train_combined = train_with_groups.iloc[train_idx][Config.target].values
    X_val_combined = train_with_groups.iloc[val_idx].drop(columns=[Config.target, 'error_group'])
    y_val_combined = train_with_groups.iloc[val_idx][Config.target].values
    
    # Get baseline values for combined data
    y_train_baseline_combined = X_train_combined['bayes_baseline'].values
    y_val_baseline_combined = X_val_combined['bayes_baseline'].values
    y_test_baseline_combined = test_with_groups['bayes_baseline'].values
    
    # Calculate residuals
    y_train_residual_combined = y_train_combined - y_train_baseline_combined
    y_val_residual_combined = y_val_combined - y_val_baseline_combined
    
    # Store residuals before log transform (fold 1 only)
    if fold == 0:
        visualization_data['residuals_before_log'] = y_train_residual_combined.copy()
    
    # Apply log transformation (variance stabilization)
    y_train_log = np.log1p(y_train_residual_combined + 1)
    y_val_log = np.log1p(y_val_residual_combined + 1)
    
    # Store residuals after log transform (fold 1 only)
    if fold == 0:
        visualization_data['residuals_after_log'] = y_train_log.copy()
    
    # Train final model
    model_combined = CatBoostRegressor(**catboost_params)
    model_combined.fit(
        X_train_combined, y_train_log,
        eval_set=(X_val_combined, y_val_log),
        verbose=False
    )
    
    # Predictions with inverse log transform
    preds_log_val = model_combined.predict(X_val_combined)
    preds_val = np.clip(np.expm1(preds_log_val) - 1 + y_val_baseline_combined, 0, 1)
    
    preds_log_test = model_combined.predict(test_with_groups)
    preds_test_fold = np.clip(np.expm1(preds_log_test) - 1 + y_test_baseline_combined, 0, 1)
    
    # Store predictions
    oof_preds[val_idx] = preds_val
    test_preds += preds_test_fold / Config.n_splits
    
    # Calculate fold score
    fold_rmse = root_mean_squared_error(y_val, preds_val)
    fold_scores.append(fold_rmse)
    print(f"\n  ✓ Fold {fold+1} RMSE: {fold_rmse:.6f}")
    print(f"  ✓ Best iteration: {model_combined.best_iteration_}")
    
    # Store feature importance from Fold 1
    if fold == 0:
        feature_importance_fold1 = pd.DataFrame({
            'feature': X_train_combined.columns,
            'importance': model_combined.get_feature_importance()
        }).sort_values('importance', ascending=False).reset_index(drop=True)

# Calculate overall CV score
cv_rmse = root_mean_squared_error(y, oof_preds)

print(f"\n{'='*80}")
print(f"CROSS-VALIDATION RESULTS")
print(f"{'='*80}")
print(f"\nFold-by-Fold Scores:")
for i, score in enumerate(fold_scores):
    print(f"  Fold {i+1}: {score:.6f}")
print(f"\n  Mean:   {np.mean(fold_scores):.6f}")
print(f"  Std:    {np.std(fold_scores):.6f}")
print(f"  Min:    {min(fold_scores):.6f}")
print(f"  Max:    {max(fold_scores):.6f}")
print(f"\n{'='*80}")
print(f"OVERALL CV RMSE: {cv_rmse:.6f}")
print(f"{'='*80}")

# ===========================
# VISUALIZATIONS
# ===========================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))

# 1. Log Transformation Effect
ax1 = plt.subplot(2, 3, 1)
before = visualization_data['residuals_before_log']
plt.hist(before, bins=50, alpha=0.7, color='coral', edgecolor='black')
plt.axvline(np.mean(before), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(before):.4f}')
plt.axvline(np.median(before), color='darkred', linestyle=':', linewidth=2, label=f'Median: {np.median(before):.4f}')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.title('Residuals BEFORE Log Transform\n(Original Scale)', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

ax2 = plt.subplot(2, 3, 2)
after = visualization_data['residuals_after_log']
plt.hist(after, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(np.mean(after), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(after):.4f}')
plt.axvline(np.median(after), color='darkblue', linestyle=':', linewidth=2, label=f'Median: {np.median(after):.4f}')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.title('Residuals AFTER Log Transform\n(Variance Stabilized)', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Error Distribution & Threshold
ax3 = plt.subplot(2, 3, 3)
errors = visualization_data['errors_distribution']
threshold = np.percentile(errors, 75)
plt.hist(errors, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axvline(threshold, color='red', linestyle='--', linewidth=2.5, 
            label=f'75th Percentile\n(Threshold: {threshold:.4f})')
plt.axvline(np.mean(errors), color='green', linestyle=':', linewidth=2, 
            label=f'Mean: {np.mean(errors):.4f}')
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.title('Error Distribution & High-Error Threshold\n(Fold 1 Validation)', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Error Groups Comparison
ax4 = plt.subplot(2, 3, 4)
error_groups = visualization_data['error_groups']
low_error_count = (error_groups == 0).sum()
high_error_count = (error_groups == 1).sum()
bars = plt.bar(['Low Error\nGroup', 'High Error\nGroup'], 
               [low_error_count, high_error_count],
               color=['lightblue', 'salmon'], edgecolor='black', linewidth=2)
plt.ylabel('Number of Samples')
plt.title('Error Group Distribution\n(75th Percentile Split)', fontweight='bold')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}\n({height/len(error_groups)*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 4. Feature Type Distribution
ax5 = plt.subplot(2, 3, 5)
feature_types = {
    'Original\nNumerical': X.shape[1] - len(cat_features) - len(group_cols) - 2,  # -2 for squared features
    'Squared\nFeatures': 2,
    'Binned\nFeatures': len([c for c in X.columns if 'tercile' in c or 'quartile' in c]),
    'Error-Group\nFeatures': len(group_cols),
    'Categorical': len(cat_features),
    'Bayesian\nBaseline': 1
}
colors_pie = ['#FF9999', '#66B2FF', '#99FF99', '#FFD700', '#FF99CC', '#B19CD9']
wedges, texts, autotexts = plt.pie(feature_types.values(), labels=feature_types.keys(), 
                                     autopct='%1.1f%%', startangle=90, colors=colors_pie,
                                     textprops={'fontsize': 9, 'fontweight': 'bold'})
for autotext in autotexts:
    autotext.set_color('white')
plt.title(f'Feature Type Distribution\n(Total: {X.shape[1]} features)', fontweight='bold')

# 5. Top 15 Most Important Features
ax6 = plt.subplot(2, 3, 6)
top_features = feature_importance_fold1.head(15).sort_values('importance', ascending=True)
colors_bar = ['#FF6B6B' if 'TE_' in f else '#4ECDC4' if 'org_' in f else '#95E1D3' 
              for f in top_features['feature']]
bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors_bar, edgecolor='black')
plt.yticks(range(len(top_features)), top_features['feature'], fontsize=8)
plt.xlabel('Importance Score')
plt.title('Top 15 Most Important Features\n(Fold 1)', fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

# Add legend for colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF6B6B', label='Error-Group Features'),
    Patch(facecolor='#4ECDC4', label='Original Target Encoding'),
    Patch(facecolor='#95E1D3', label='Other Features')
]
plt.legend(handles=legend_elements, loc='lower right', fontsize=7)

plt.tight_layout()
plt.savefig('model_analysis_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_analysis_visualization.png")
plt.show()

# ===========================
# FEATURE IMPORTANCE TABLE (TOP 30)
# ===========================
print("\n" + "="*80)
print("FEATURE IMPORTANCE (FOLD 1 - TOP 30)")
print("="*80)

print(f"\nTotal features: {len(feature_importance_fold1)}")
print("\nTop 30 Most Important Features:")
print("="*70)
for idx, row in feature_importance_fold1.head(30).iterrows():
    feature_type = ''
    if 'TE_' in row['feature']:
        feature_type = ' [ERROR-GROUP]'
    elif 'org_' in row['feature']:
        feature_type = ' [ORIG-TE]'
    elif 'tercile' in row['feature'] or 'quartile' in row['feature']:
        feature_type = ' [BINNED]'
    elif '_sq' in row['feature']:
        feature_type = ' [SQUARED]'
    elif row['feature'] == 'bayes_baseline':
        feature_type = ' [BAYESIAN]'
    
    print(f"{idx+1:3d}. {row['feature']:45s} {row['importance']:10.2f}{feature_type}")

# ===========================
# SAVE OUTPUTS
# ===========================
print("\n" + "="*80)
print("SAVING PREDICTIONS AND OUTPUTS")
print("="*80)

# Generate timestamp and filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = "CatBoost_Optimized"
cv_score_str = f"{cv_rmse:.5f}".replace(".", "")

# Save OOF predictions
oof_filename = f"oof_{model_name}_{timestamp}_CV0{cv_score_str}.npy"
np.save(oof_filename, oof_preds)
print(f"✓ OOF predictions saved: {oof_filename}")

# Save test predictions
test_filename = f"test_{model_name}_{timestamp}_CV0{cv_score_str}.npy"
np.save(test_filename, test_preds)
print(f"✓ Test predictions saved: {test_filename}")

# Save feature importance
importance_filename = f"feature_importance_{model_name}_{timestamp}_CV0{cv_score_str}.csv"
feature_importance_fold1.to_csv(importance_filename, index=False)
print(f"✓ Feature importance saved: {importance_filename}")

# Save submission file
submission = pd.DataFrame({
    'id': test.index,
    Config.target: test_preds
})

# Save timestamped version for your records
backup_filename = f"submission_{model_name}_{timestamp}_CV0{cv_score_str}.csv"
submission.to_csv(backup_filename, index=False)
print(f"✓ Backup submission saved: {backup_filename}")

# Save clean version for Kaggle upload
submission.to_csv("submission.csv", index=False)
print(f"✓ Kaggle submission saved: submission.csv")

# ===========================
# PREDICTION STATISTICS
# ===========================
print("\n" + "="*80)
print("PREDICTION STATISTICS")
print("="*80)
print(f"\nOOF Predictions:")
print(f"  Mean:  {oof_preds.mean():.6f}")
print(f"  Std:   {oof_preds.std():.6f}")
print(f"  Min:   {oof_preds.min():.6f}")
print(f"  Max:   {oof_preds.max():.6f}")
print(f"  Q25:   {np.percentile(oof_preds, 25):.6f}")
print(f"  Q50:   {np.percentile(oof_preds, 50):.6f}")
print(f"  Q75:   {np.percentile(oof_preds, 75):.6f}")

print(f"\nTest Predictions:")
print(f"  Mean:  {test_preds.mean():.6f}")
print(f"  Std:   {test_preds.std():.6f}")
print(f"  Min:   {test_preds.min():.6f}")
print(f"  Max:   {test_preds.max():.6f}")
print(f"  Q25:   {np.percentile(test_preds, 25):.6f}")
print(f"  Q50:   {np.percentile(test_preds, 50):.6f}")
print(f"  Q75:   {np.percentile(test_preds, 75):.6f}")

# ===========================
# FINAL SUMMARY
# ===========================
print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nModel Configuration:")
print(f"  Algorithm:        CatBoost Regressor (Optimized)")
print(f"  Learning Rate:    {catboost_params['learning_rate']:.6f}")
print(f"  Depth:            {catboost_params['depth']}")
print(f"  Grow Policy:      {catboost_params['grow_policy']}")
print(f"  Folds:            {Config.n_splits}")
print(f"  Task:             GPU-accelerated")
print(f"\nData:")
print(f"  Train samples:    {len(train):,}")
print(f"  Test samples:     {len(test):,}")
print(f"  Original data:    112,000 samples (2k + 10k + 100k)")
print(f"  Final features:   {X.shape[1]}")
print(f"\nPerformance:")
print(f"  CV RMSE:          {cv_rmse:.6f}")
print(f"  Best fold:        {min(fold_scores):.6f}")
print(f"  Worst fold:       {max(fold_scores):.6f}")
print(f"  Std deviation:    {np.std(fold_scores):.6f}")
print(f"\nKey Feature Engineering:")
print(f"  ✓ Curvature TERCILES (3 bins)")
print(f"  ✓ Other features QUARTILES (4 bins)")
print(f"  ✓ Bayesian baseline (residual learning)")
print(f"  ✓ Variance stabilization (log transform)")
print(f"  ✓ Error-group target encoding (12 features)")
print(f"  ✓ Original data target encoding (mean/median)")
print(f"  ✓ Squared features (curvature, speed_limit)")
print(f"  ✓ Predictions clipped to [0, 1]")
print(f"\nFiles Created:")
print(f"  → {oof_filename}")
print(f"  → {test_filename}")
print(f"  → {importance_filename}")
print(f"  → {backup_filename}")  # ✓ Fixed
print(f"  → submission.csv")  # You can also add this for clarity
print(f"  → model_analysis_visualization.png")
print("="*80)

