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


"""
Diamond Price Prediction - LightGBM Solution
==============================================
Model: LightGBM with optimized hyperparameters
CV Strategy: 15-fold cross-validation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

# =========================
# Configuration
# =========================
SEED = 42
np.random.seed(SEED)

# Optimized hyperparameters to maximize R²
BEST_PARAMS = {
    'n_estimators': 1551,
    'learning_rate': 0.053815213345167076,
    'num_leaves': 44,
    'max_depth': 3,
    'min_child_samples': 16,
    'feature_fraction': 0.735983691316846,
    'bagging_fraction': 0.7056526109599511,
    'bagging_freq': 6,
    'lambda_l1': 0.020499490790283646,
    'lambda_l2': 0.7382366712998285,
    'min_gain_to_split': 0.0652875198904059,
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'random_state': SEED,
    'n_jobs': -1,
    'verbosity': -1
}

# =========================
# Load Data
# =========================
print("="*70)
print("DIAMOND PRICE PREDICTION - LIGHTGBM SOLUTION")
print("="*70)
print("\nLoading data...")

train0 = pd.read_csv("/kaggle/input/predicting-the-price-of-diamond/train.csv")
test0 = pd.read_csv("/kaggle/input/predicting-the-price-of-diamond/test.csv")
submit = pd.read_csv('/kaggle/input/predicting-the-price-of-diamond/submission.csv')

print(f"Train shape: {train0.shape}")
print(f"Test shape:  {test0.shape}")

# =========================
# Feature Engineering
# =========================
def get_base_encodings(df):
    """Ordinal encoding for categorical features"""
    cut_mapping = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
    color_mapping = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
    clarity_mapping = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}
    
    encodings = {}
    if 'cut' in df.columns:
        encodings['cut_encoded'] = df['cut'].map(cut_mapping).fillna(3)
    if 'color' in df.columns:
        encodings['color_encoded'] = df['color'].map(color_mapping).fillna(4)
    if 'clarity' in df.columns:
        encodings['clarity_encoded'] = df['clarity'].map(clarity_mapping).fillna(4)
    
    return encodings

def feature_set_13(df):
    """Statistical aggregates - Feature Set 13"""
    df = df.copy()
    
    # Volume feature
    df['vol'] = df['x'] * df['y'] * df['z']
    
    # Dimensional statistics
    dims = df[['x', 'y', 'z']]
    df['dim_mean'] = dims.mean(axis=1)
    df['dim_std'] = dims.std(axis=1)
    df['dim_median'] = dims.median(axis=1)
    df['dim_cv'] = df['dim_std'] / (df['dim_mean'] + 0.001)
    
    # Ordinal encodings
    encodings = get_base_encodings(df)
    for col, vals in encodings.items():
        df[col] = vals
    
    # Drop original categorical columns
    df.drop(columns=[c for c in ['cut', 'color', 'clarity', 'id'] if c in df.columns], inplace=True)
    
    return df.fillna(0).replace([np.inf, -np.inf], 0)

# =========================
# Prepare Data
# =========================
print("\nEngineering features...")
train_fe = feature_set_13(train0.copy())
test_fe = feature_set_13(test0.copy())

y = train_fe['price']
X = train_fe.drop(columns=['price'])

print(f"Training features: {X.shape}")
print(f"Test features:     {test_fe.shape}")
print(f"Feature list:      {list(X.columns)}")

# Standardize features
print("\nScaling features...")
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
test_scaled = pd.DataFrame(scaler.transform(test_fe), columns=test_fe.columns)

# =========================
# Train Model with 15-Fold CV
# =========================
print("\n" + "="*70)
print("TRAINING LIGHTGBM WITH OPTIMIZED HYPERPARAMETERS")
print("="*70)
print("\nModel Configuration:")
print(f"  • Estimators:        {BEST_PARAMS['n_estimators']}")
print(f"  • Learning Rate:     {BEST_PARAMS['learning_rate']:.6f}")
print(f"  • Num Leaves:        {BEST_PARAMS['num_leaves']}")
print(f"  • Max Depth:         {BEST_PARAMS['max_depth']}")
print(f"  • Feature Fraction:  {BEST_PARAMS['feature_fraction']:.4f}")
print(f"  • Bagging Fraction:  {BEST_PARAMS['bagging_fraction']:.4f}")
print(f"  • CV Folds:          15")
print("\n" + "-"*70)

kf = KFold(n_splits=15, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(y))
test_preds = np.zeros(len(test_scaled))
models = []
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
    X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train model
    model = lgb.LGBMRegressor(**BEST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0)
        ]
    )
    
    # Generate predictions
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(test_scaled) / 15
    
    # Calculate fold score
    fold_r2 = r2_score(y_val, oof_preds[val_idx])
    fold_scores.append(fold_r2)
    
    print(f"Fold {fold:2d}/15 | R² = {fold_r2:.6f} | Trees used: {model.best_iteration_}")
    
    models.append(model)

# =========================
# Evaluation Results
# =========================
overall_r2 = r2_score(y, oof_preds)
rmse = np.sqrt(mean_squared_error(y, oof_preds))
mae = mean_absolute_error(y, oof_preds)
residuals = y - oof_preds

print("\n" + "="*70)
print("CROSS-VALIDATION RESULTS")
print("="*70)
print(f"Mean R²:     {np.mean(fold_scores):.6f}")
print(f"Std R²:      {np.std(fold_scores):.6f}")
print(f"Min R²:      {np.min(fold_scores):.6f}")
print(f"Max R²:      {np.max(fold_scores):.6f}")
print(f"OOF R²:      {overall_r2:.6f}")
print(f"RMSE:        {rmse:.2f}")
print(f"MAE:         {mae:.2f}")
print("="*70)

# =========================
# Residual Analysis Visualizations
# =========================
print("\n" + "="*70)
print("RESIDUAL ANALYSIS")
print("="*70)

fig = plt.figure(figsize=(20, 12))

# 1. Predicted vs Actual
ax1 = plt.subplot(2, 3, 1)
plt.scatter(y, oof_preds, alpha=0.5, s=10, edgecolors='none')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price', fontsize=12, fontweight='bold')
plt.ylabel('Predicted Price', fontsize=12, fontweight='bold')
plt.title(f'Predicted vs Actual\nR² = {overall_r2:.6f}', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Residuals vs Predicted
ax2 = plt.subplot(2, 3, 2)
plt.scatter(oof_preds, residuals, alpha=0.5, s=10, edgecolors='none')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Price', fontsize=12, fontweight='bold')
plt.ylabel('Residuals', fontsize=12, fontweight='bold')
plt.title('Residual Plot\n(Random scatter = good fit)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# 3. Residual Distribution
ax3 = plt.subplot(2, 3, 3)
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Residual')
plt.xlabel('Residuals', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title(f'Residual Distribution\nMean = {residuals.mean():.2f}', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 4. Q-Q Plot
ax4 = plt.subplot(2, 3, 4)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot\n(Tests normality of residuals)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# 5. Residuals vs Actual
ax5 = plt.subplot(2, 3, 5)
plt.scatter(y, residuals, alpha=0.5, s=10, edgecolors='none')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Actual Price', fontsize=12, fontweight='bold')
plt.ylabel('Residuals', fontsize=12, fontweight='bold')
plt.title('Residuals vs Actual Price', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# 6. Absolute Residuals vs Predicted
ax6 = plt.subplot(2, 3, 6)
plt.scatter(oof_preds, np.abs(residuals), alpha=0.5, s=10, edgecolors='none')
plt.xlabel('Predicted Price', fontsize=12, fontweight='bold')
plt.ylabel('Absolute Residuals', fontsize=12, fontweight='bold')
plt.title('Absolute Residuals vs Predicted\n(Check for heteroscedasticity)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Residual Statistics
print("\nResidual Statistics:")
print(f"  Mean:              {residuals.mean():.4f}")
print(f"  Std Dev:           {residuals.std():.4f}")
print(f"  Min:               {residuals.min():.4f}")
print(f"  Max:               {residuals.max():.4f}")
print(f"  25th Percentile:   {np.percentile(residuals, 25):.4f}")
print(f"  Median:            {np.median(residuals):.4f}")
print(f"  75th Percentile:   {np.percentile(residuals, 75):.4f}")

# Outlier Analysis
outlier_threshold = 3 * residuals.std()
outliers = np.abs(residuals) > outlier_threshold
print(f"\nOutlier Analysis (|residual| > 3σ):")
print(f"  Number of outliers: {outliers.sum()} ({100*outliers.sum()/len(residuals):.2f}%)")
print(f"  Threshold:          ±{outlier_threshold:.2f}")

# Normality Test
_, p_value = stats.shapiro(residuals[:5000])  # Shapiro test on sample
print(f"\nNormality Test (Shapiro-Wilk):")
print(f"  p-value:            {p_value:.6f}")
print(f"  Normal distribution: {'Yes (p > 0.05)' if p_value > 0.05 else 'No (p < 0.05)'}")

print("="*70)
print("\n✓ Residual analysis plots saved as 'residual_analysis.png'")
print("="*70)

# =========================
# Feature Importance
# =========================
print("\n" + "="*70)
print("TOP 10 FEATURE IMPORTANCES")
print("="*70)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': np.mean([m.feature_importances_ for m in models], axis=0)
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']:20s} {row['importance']:>10.1f}")

print("="*70)

# =========================
# Create Submission
# =========================
submission = submit.copy()
submission['price'] = test_preds

filename = 'submission.csv'
submission.to_csv(filename, index=False)

print(f"\n✅ SUCCESS!")
print(f"\nSubmission file created: {filename}")
print(f"Out-of-fold R² Score:    {overall_r2:.6f}")
print("\n" + "="*70)

