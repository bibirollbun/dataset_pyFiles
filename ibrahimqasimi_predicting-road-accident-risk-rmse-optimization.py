import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Consistent visualization theme
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'danger': '#D62246',
    'neutral': '#6C757D'
}

PALETTE = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], 
           COLORS['success'], COLORS['danger']]

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(PALETTE)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 4)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Environment configured successfully")
print(f"XGBoost: {xgb.__version__}")
print(f"LightGBM: {lgb.__version__}")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission_template = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

print(f"Training set: {train_df.shape}")
print(f"Test set: {test_df.shape}")
print(f"Memory usage: {train_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


# Data quality checks
print("Data Quality Report")
print("=" * 60)
print(f"Missing values: {train_df.isnull().sum().sum()}")
print(f"Duplicate rows: {train_df.duplicated().sum()}")
print(f"\nData types:")
print(train_df.dtypes.value_counts())

# Define feature groups
CATEGORICAL_FEATURES = ['road_type', 'lighting', 'weather', 'time_of_day']
BOOLEAN_FEATURES = ['road_signs_present', 'public_road', 'holiday', 'school_season']
NUMERICAL_FEATURES = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
TARGET = 'accident_risk'
ID_COL = 'id'


# Target distribution analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor('white')

# Histogram
axes[0].hist(train_df[TARGET], bins=50, color=COLORS['primary'], 
             edgecolor='black', alpha=0.7, linewidth=1.2)
axes[0].axvline(train_df[TARGET].mean(), color=COLORS['danger'], 
                linestyle='--', linewidth=2.5, label=f'Mean: {train_df[TARGET].mean():.3f}')
axes[0].axvline(train_df[TARGET].median(), color=COLORS['success'], 
                linestyle='--', linewidth=2.5, label=f'Median: {train_df[TARGET].median():.3f}')
axes[0].set_xlabel('Accident Risk', fontweight='bold')
axes[0].set_ylabel('Frequency', fontweight='bold')
axes[0].set_title('Target Distribution', fontweight='bold', pad=15)
axes[0].legend(frameon=True, shadow=True)
axes[0].grid(alpha=0.3, linestyle='--')

# Boxplot
bp = axes[1].boxplot(train_df[TARGET], vert=True, patch_artist=True,
                     boxprops=dict(facecolor=COLORS['primary'], alpha=0.7, linewidth=1.5),
                     medianprops=dict(color=COLORS['danger'], linewidth=2.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5),
                     flierprops=dict(marker='o', markerfacecolor=COLORS['accent'], 
                                    markersize=6, alpha=0.5))
axes[1].set_ylabel('Accident Risk', fontweight='bold')
axes[1].set_title('Distribution Statistics', fontweight='bold', pad=15)
axes[1].grid(alpha=0.3, linestyle='--')

# Q-Q Plot
stats.probplot(train_df[TARGET], dist="norm", plot=axes[2])
axes[2].get_lines()[0].set_color(COLORS['primary'])
axes[2].get_lines()[0].set_markersize(4)
axes[2].get_lines()[0].set_alpha(0.6)
axes[2].get_lines()[1].set_color(COLORS['danger'])
axes[2].get_lines()[1].set_linewidth(2)
axes[2].set_title('Q-Q Plot', fontweight='bold', pad=15)
axes[2].grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('/kaggle/working/target_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print(f"Target Statistics:")
print(f"Mean: {train_df[TARGET].mean():.4f}")
print(f"Median: {train_df[TARGET].median():.4f}")
print(f"Std: {train_df[TARGET].std():.4f}")
print(f"Skewness: {train_df[TARGET].skew():.4f}")
print(f"Range: [{train_df[TARGET].min():.4f}, {train_df[TARGET].max():.4f}]")


# Numerical features correlation
numerical_data = train_df[NUMERICAL_FEATURES + [TARGET]]
correlation_matrix = numerical_data.corr()

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.patch.set_facecolor('white')

# Correlation heatmap
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', 
            cmap='RdYlBu_r', center=0, square=True, 
            linewidths=2, cbar_kws={"shrink": 0.8},
            annot_kws={'size': 11, 'weight': 'bold'},
            ax=axes[0])
axes[0].set_title('Correlation Matrix', fontweight='bold', pad=15, fontsize=14)

# Target correlation bar plot
target_corr = correlation_matrix[TARGET].drop(TARGET).sort_values()
colors_bar = [COLORS['success'] if x > 0 else COLORS['danger'] for x in target_corr]
bars = axes[1].barh(range(len(target_corr)), target_corr.values, 
                    color=colors_bar, edgecolor='black', linewidth=1.5, alpha=0.8)
axes[1].set_yticks(range(len(target_corr)))
axes[1].set_yticklabels(target_corr.index, fontweight='bold')
axes[1].set_xlabel('Correlation Coefficient', fontweight='bold')
axes[1].set_title('Feature Correlation with Target', fontweight='bold', pad=15, fontsize=14)
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
axes[1].grid(alpha=0.3, linestyle='--', axis='x')

for i, (bar, val) in enumerate(zip(bars, target_corr.values)):
    x_pos = val + (0.02 if val > 0 else -0.02)
    ha = 'left' if val > 0 else 'right'
    axes[1].text(x_pos, i, f'{val:.3f}', va='center', ha=ha, 
                fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('/kaggle/working/correlation_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# Categorical features analysis
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.patch.set_facecolor('white')
axes = axes.ravel()

for idx, feature in enumerate(CATEGORICAL_FEATURES):
    category_stats = train_df.groupby(feature)[TARGET].agg(['mean', 'std', 'count']).sort_values('mean')
    
    # Create gradient colors
    n_cats = len(category_stats)
    colors_gradient = [COLORS['success'] if i < n_cats/2 else COLORS['danger'] 
                      for i in range(n_cats)]
    
    bars = axes[idx].barh(range(len(category_stats)), category_stats['mean'],
                          xerr=category_stats['std'], 
                          color=colors_gradient, alpha=0.8, 
                          edgecolor='black', linewidth=1.5,
                          error_kw={'linewidth': 2, 'elinewidth': 1.5})
    
    axes[idx].set_yticks(range(len(category_stats)))
    axes[idx].set_yticklabels(category_stats.index, fontweight='bold')
    axes[idx].set_xlabel('Mean Accident Risk', fontweight='bold')
    axes[idx].set_title(f'{feature.replace("_", " ").title()}', 
                       fontweight='bold', pad=15, fontsize=13)
    axes[idx].grid(alpha=0.3, linestyle='--', axis='x')
    
    for i, (val, count) in enumerate(zip(category_stats['mean'], category_stats['count'])):
        axes[idx].text(val + 0.01, i, f'{val:.3f}\n(n={count:,})', 
                      va='center', fontsize=9, fontweight='bold')

plt.suptitle('Categorical Feature Impact Analysis', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('/kaggle/working/categorical_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


def engineer_features(df):
    """
    Create domain-informed feature interactions.
    """
    df_eng = df.copy()
    
    # Core interactions
    df_eng['curv_speed'] = df_eng['curvature'] * df_eng['speed_limit']
    df_eng['lane_speed'] = df_eng['num_lanes'] * df_eng['speed_limit']
    df_eng['accidents_speed'] = df_eng['num_reported_accidents'] * df_eng['speed_limit']
    df_eng['accidents_curv'] = df_eng['num_reported_accidents'] * df_eng['curvature']
    
    # Polynomial features
    df_eng['curvature_sq'] = df_eng['curvature'] ** 2
    df_eng['curvature_cube'] = df_eng['curvature'] ** 3
    df_eng['speed_sq'] = df_eng['speed_limit'] ** 2
    
    # Risk scores
    df_eng['risk_intensity'] = (df_eng['curvature'] * df_eng['speed_limit']) / 50
    df_eng['lane_capacity_risk'] = (5 - df_eng['num_lanes']) * df_eng['speed_limit']
    df_eng['accidents_per_lane'] = df_eng['num_reported_accidents'] / (df_eng['num_lanes'] + 1)
    
    # Binary indicators
    df_eng['high_risk_combo'] = ((df_eng['curvature'] > 0.5) & 
                                  (df_eng['speed_limit'] >= 60)).astype(int)
    
    return df_eng

# Preprocessing
train_processed = train_df.copy()
test_processed = test_df.copy()

# Convert booleans
for col in BOOLEAN_FEATURES:
    train_processed[col] = train_processed[col].astype(int)
    test_processed[col] = test_processed[col].astype(int)

# Label encode categoricals
label_encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    train_processed[f'{col}_enc'] = le.fit_transform(train_processed[col])
    test_processed[f'{col}_enc'] = le.transform(test_processed[col])
    label_encoders[col] = le

# Apply feature engineering
train_engineered = engineer_features(train_processed)
test_engineered = engineer_features(test_processed)

print(f"Feature engineering complete")
print(f"Original features: {len(CATEGORICAL_FEATURES + BOOLEAN_FEATURES + NUMERICAL_FEATURES)}")
print(f"Engineered features: {train_engineered.shape[1]}")
print(f"New features created: {train_engineered.shape[1] - train_processed.shape[1]}")


# Prepare feature matrix
exclude_cols = [ID_COL, TARGET] + CATEGORICAL_FEATURES
feature_cols = [col for col in train_engineered.columns if col not in exclude_cols]

X_train = train_engineered[feature_cols].values
y_train = train_engineered[TARGET].values
X_test = test_engineered[feature_cols].values

print(f"Training matrix: {X_train.shape}")
print(f"Test matrix: {X_test.shape}")


# Cross-validation setup
N_SPLITS = 5
kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Storage for predictions
oof_xgb = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))

test_xgb = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

models_xgb, models_lgb, models_cat = [], [], []
scores_xgb, scores_lgb, scores_cat = [], [], []


# XGBoost with GPU
xgb_params = {
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'max_depth': 8,
    'min_child_weight': 2,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

print("Training XGBoost with GPU acceleration")
print("=" * 60)

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    print(f"Fold {fold_idx}/{N_SPLITS}", end=" ")
    
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        early_stopping_rounds=100,
        verbose=False
    )
    
    oof_xgb[val_idx] = model.predict(X_fold_val)
    test_xgb += model.predict(X_test) / N_SPLITS
    
    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, oof_xgb[val_idx]))
    scores_xgb.append(fold_rmse)
    models_xgb.append(model)
    
    print(f"RMSE: {fold_rmse:.6f} | Best iter: {model.best_iteration}")

xgb_oof_rmse = np.sqrt(mean_squared_error(y_train, oof_xgb))
print(f"\nXGBoost OOF RMSE: {xgb_oof_rmse:.6f}")
print(f"CV Std: {np.std(scores_xgb):.6f}")
print("=" * 60)


# LightGBM with GPU
lgb_params = {
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'num_leaves': 63,
    'max_depth': 8,
    'min_child_samples': 20,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'verbose': -1
}

print("Training LightGBM with GPU acceleration")
print("=" * 60)

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    print(f"Fold {fold_idx}/{N_SPLITS}", end=" ")
    
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    oof_lgb[val_idx] = model.predict(X_fold_val)
    test_lgb += model.predict(X_test) / N_SPLITS
    
    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, oof_lgb[val_idx]))
    scores_lgb.append(fold_rmse)
    models_lgb.append(model)
    
    print(f"RMSE: {fold_rmse:.6f} | Best iter: {model.best_iteration_}")

lgb_oof_rmse = np.sqrt(mean_squared_error(y_train, oof_lgb))
print(f"\nLightGBM OOF RMSE: {lgb_oof_rmse:.6f}")
print(f"CV Std: {np.std(scores_lgb):.6f}")
print("=" * 60)


# CatBoost with GPU
cat_params = {
    'iterations': 2000,
    'learning_rate': 0.02,
    'depth': 8,
    'l2_leaf_reg': 3,
    'random_state': RANDOM_STATE,
    'task_type': 'GPU',
    'devices': '0',
    'verbose': 0,
    'early_stopping_rounds': 100
}

print("Training CatBoost with GPU acceleration")
print("=" * 60)

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    print(f"Fold {fold_idx}/{N_SPLITS}", end=" ")
    
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
    
    model = CatBoostRegressor(**cat_params)
    model.fit(
        X_fold_train, y_fold_train,
        eval_set=(X_fold_val, y_fold_val),
        verbose=False
    )
    
    oof_cat[val_idx] = model.predict(X_fold_val)
    test_cat += model.predict(X_test) / N_SPLITS
    
    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, oof_cat[val_idx]))
    scores_cat.append(fold_rmse)
    models_cat.append(model)
    
    print(f"RMSE: {fold_rmse:.6f} | Best iter: {model.best_iteration_}")

cat_oof_rmse = np.sqrt(mean_squared_error(y_train, oof_cat))
print(f"\nCatBoost OOF RMSE: {cat_oof_rmse:.6f}")
print(f"CV Std: {np.std(scores_cat):.6f}")
print("=" * 60)


# Model performance comparison
results_df = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'CatBoost'],
    'OOF RMSE': [xgb_oof_rmse, lgb_oof_rmse, cat_oof_rmse],
    'CV Std': [np.std(scores_xgb), np.std(scores_lgb), np.std(scores_cat)]
}).sort_values('OOF RMSE')

print("\nModel Performance Summary")
print("=" * 60)
print(results_df.to_string(index=False))
print("=" * 60)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.patch.set_facecolor('white')

# Bar chart
bars = axes[0].barh(results_df['Model'], results_df['OOF RMSE'], 
                    color=[COLORS['primary'], COLORS['success'], COLORS['accent']], 
                    alpha=0.8, edgecolor='black', linewidth=2)
axes[0].set_xlabel('Out-of-Fold RMSE', fontweight='bold', fontsize=12)
axes[0].set_title('Model Performance Comparison', fontweight='bold', fontsize=14, pad=15)
axes[0].grid(alpha=0.3, linestyle='--', axis='x')

for bar, val in zip(bars, results_df['OOF RMSE']):
    axes[0].text(val + 0.00005, bar.get_y() + bar.get_height()/2, 
                f'{val:.6f}', va='center', fontweight='bold', fontsize=11)

# Box plot
fold_data = pd.DataFrame({
    'XGBoost': scores_xgb,
    'LightGBM': scores_lgb,
    'CatBoost': scores_cat
})

bp = axes[1].boxplot([scores_xgb, scores_lgb, scores_cat], 
                     labels=['XGBoost', 'LightGBM', 'CatBoost'],
                     patch_artist=True,
                     boxprops=dict(facecolor=COLORS['primary'], alpha=0.7, linewidth=1.5),
                     medianprops=dict(color=COLORS['danger'], linewidth=2.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))

for patch, color in zip(bp['boxes'], [COLORS['primary'], COLORS['success'], COLORS['accent']]):
    patch.set_facecolor(color)
    
axes[1].set_ylabel('RMSE', fontweight='bold', fontsize=12)
axes[1].set_title('Cross-Validation Stability', fontweight='bold', fontsize=14, pad=15)
axes[1].grid(alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
plt.savefig('/kaggle/working/model_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# Weighted ensemble based on inverse RMSE
weights = 1 / np.array([xgb_oof_rmse, lgb_oof_rmse, cat_oof_rmse])
weights = weights / weights.sum()

print("Ensemble Weights (Inverse RMSE):")
print(f"  XGBoost: {weights[0]:.4f}")
print(f"  LightGBM: {weights[1]:.4f}")
print(f"  CatBoost: {weights[2]:.4f}")

# Calculate ensemble predictions
ensemble_oof = weights[0] * oof_xgb + weights[1] * oof_lgb + weights[2] * oof_cat
ensemble_test = weights[0] * test_xgb + weights[1] * test_lgb + weights[2] * test_cat

# Clip predictions
ensemble_test = np.clip(ensemble_test, 0, 1)

# Calculate ensemble RMSE
ensemble_rmse = np.sqrt(mean_squared_error(y_train, ensemble_oof))

print(f"\nEnsemble OOF RMSE: {ensemble_rmse:.6f}")
print(f"Improvement vs best single model: {(results_df['OOF RMSE'].min() - ensemble_rmse)*10000:.2f} bps")


# Feature importance from best model
best_model = models_lgb[0]

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

# Plot top 20 features
fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor('white')

top_n = 20
top_features = feature_importance.head(top_n)
colors_gradient = plt.cm.RdYlBu_r(np.linspace(0.2, 0.9, top_n))

bars = ax.barh(range(top_n), top_features['importance'], 
               color=colors_gradient, edgecolor='black', linewidth=1.5, alpha=0.8)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features['feature'], fontweight='bold')
ax.set_xlabel('Importance Score', fontweight='bold', fontsize=12)
ax.set_title(f'Top {top_n} Feature Importances', fontweight='bold', fontsize=14, pad=15)
ax.invert_yaxis()
ax.grid(alpha=0.3, linestyle='--', axis='x')

for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
    ax.text(val + max(top_features['importance'])*0.01, i, f'{val:.0f}', 
            va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('/kaggle/working/feature_importance.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

feature_importance.to_csv('/kaggle/working/feature_importance.csv', index=False)


# Prediction analysis
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.patch.set_facecolor('white')

# Scatter plot
axes[0].scatter(y_train, ensemble_oof, alpha=0.4, s=8, 
               c=COLORS['primary'], edgecolors='none', rasterized=True)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
            color=COLORS['danger'], linestyle='--', linewidth=3, label='Perfect Prediction')
axes[0].set_xlabel('Actual Accident Risk', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Predicted Accident Risk', fontweight='bold', fontsize=12)
axes[0].set_title('Predictions vs Actuals', fontweight='bold', fontsize=14, pad=15)
axes[0].legend(frameon=True, shadow=True, fontsize=11)
axes[0].grid(alpha=0.3, linestyle='--')

# Distribution comparison
axes[1].hist(y_train, bins=50, alpha=0.6, label='Training', 
            color=COLORS['primary'], edgecolor='black', linewidth=1.2)
axes[1].hist(ensemble_test, bins=50, alpha=0.6, label='Predictions', 
            color=COLORS['accent'], edgecolor='black', linewidth=1.2)
axes[1].axvline(y_train.mean(), color=COLORS['primary'], linestyle='--', 
               linewidth=2.5, label=f'Train Mean: {y_train.mean():.3f}')
axes[1].axvline(ensemble_test.mean(), color=COLORS['accent'], linestyle='--', 
               linewidth=2.5, label=f'Pred Mean: {ensemble_test.mean():.3f}')
axes[1].set_xlabel('Accident Risk', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Frequency', fontweight='bold', fontsize=12)
axes[1].set_title('Distribution Comparison', fontweight='bold', fontsize=14, pad=15)
axes[1].legend(frameon=True, shadow=True, fontsize=10)
axes[1].grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('/kaggle/working/prediction_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'accident_risk': ensemble_test
})

# Validation
assert submission.shape[0] == test_df.shape[0]
assert submission['accident_risk'].isna().sum() == 0
assert (submission['accident_risk'] >= 0).all()
assert (submission['accident_risk'] <= 1).all()

submission.to_csv('/kaggle/working/submission.csv', index=False)

print("Submission Created Successfully")
print("=" * 60)
print(f"Shape: {submission.shape}")
print(f"\nPrediction Statistics:")
print(f"  Mean: {submission['accident_risk'].mean():.4f}")
print(f"  Std: {submission['accident_risk'].std():.4f}")
print(f"  Min: {submission['accident_risk'].min():.4f}")
print(f"  Max: {submission['accident_risk'].max():.4f}")
print(f"\nFirst 10 predictions:")
print(submission.head(10))


import os

def hierarchical_blend(paths, weights, sub_weights, desc_w=0.70, asc_w=0.30):
    """
    Hierarchical blending with position-dependent weights.
    """
    dfs = [pd.read_csv(p) for p in paths]
    
    merged = dfs[0][['id']].copy()
    for i, df in enumerate(dfs):
        merged[f'm{i}'] = df['accident_risk'].values
    
    cols = [f'm{i}' for i in range(len(dfs))]
    
    def blend_row(row, reverse):
        vals = [(i, row[c]) for i, c in enumerate(cols)]
        sorted_vals = sorted(vals, key=lambda x: x[1], reverse=reverse)
        return sum(v * (weights[i] + sub_weights[pos]) 
                  for pos, (i, v) in enumerate(sorted_vals))
    
    merged['desc'] = merged.apply(lambda r: blend_row(r, True), axis=1)
    merged['asc'] = merged.apply(lambda r: blend_row(r, False), axis=1)
    merged['accident_risk'] = desc_w * merged['desc'] + asc_w * merged['asc']
    merged['accident_risk'] = merged['accident_risk'].clip(0, 1)
    
    return merged[['id', 'accident_risk']]

# Use the correct dataset path
blend_input_path = '/kaggle/input/1-october-2025-ps-s5-e10'
use_blending = os.path.exists(blend_input_path)

if use_blending:
    print("Blending dataset found. Applying hierarchical blend...")
    
    # Use your actual submission files
    submission_paths = [
        blend_input_path + '/submission_0.05552.a.csv',
        blend_input_path + '/submission_0.05553.e.csv',
        blend_input_path + '/submission_0.05554.b.csv',
        blend_input_path + '/submission_0.05554.d.csv'
    ]
    
    # Configure weights (adjust these based on performance)
    base_weights = [0.50, 0.30, 0.10, 0.10]
    position_weights = [+0.07, +0.04, -0.04, -0.07]
    
    # Create blended submission
    blended = hierarchical_blend(
        paths=submission_paths,
        weights=base_weights,
        sub_weights=position_weights,
        desc_w=0.70,
        asc_w=0.30
    )
    
    # Save blended submission
    blended.to_csv('/kaggle/working/submission_blended.csv', index=False)
    blended.to_csv('/kaggle/working/submission.csv', index=False)
    
    print("Blended submission created successfully")
    print("\nStatistics:")
    print(f"  Mean: {blended['accident_risk'].mean():.5f}")
    print(f"  Std: {blended['accident_risk'].std():.5f}")
    print(f"  Min: {blended['accident_risk'].min():.5f}")
    print(f"  Max: {blended['accident_risk'].max():.5f}")
    print("\nFiles saved:")
    print("  - /kaggle/working/submission_blended.csv")
    print("  - /kaggle/working/submission.csv (main)")
    
    display(blended.head(10))
    
else:
    print("Blending dataset not found. Using ensemble submission.")
    print("\nMain submission already saved: /kaggle/working/submission.csv")

print("\nNotebook execution complete.")

