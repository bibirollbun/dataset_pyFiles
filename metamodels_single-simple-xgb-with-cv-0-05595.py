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
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("="*80)
print("XGBOOST BEST PARAMETERS - COMPREHENSIVE ANALYSIS")
print("="*80)

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
train.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)

# Feature Engineering
def create_frequency_features(train_df, test_df):
    train_new, test_new = train_df.copy(), test_df.copy()
    num_cols = train_new.select_dtypes(include=np.number).columns.tolist()
    cat_cols = train_new.select_dtypes(include=["object", "bool"]).columns.tolist()
    
    encoders = {}
    for col in cat_cols:
        encoders[col] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        train_new[col] = encoders[col].fit_transform(train_new[[col]])
        test_new[col] = encoders[col].transform(test_new[[col]])
    
    all_cols = train_new.columns.tolist()
    for col in all_cols:
        freq = train_new[col].value_counts(normalize=True)
        train_new[f'{col}_freq'] = train_new[col].map(freq)
        test_new[f'{col}_freq'] = test_new[col].map(freq).fillna(train_new[f'{col}_freq'].mean())
        
        if col in num_cols:
            try:
                train_new[f'{col}_bin5'] = pd.qcut(train_new[col], q=5, labels=False, duplicates='drop')
                _, bins5 = pd.qcut(train_new[col], q=5, retbins=True, duplicates='drop')
                test_new[f'{col}_bin5'] = pd.cut(test_new[col], bins=bins5, labels=False, include_lowest=True)
            except: train_new[f'{col}_bin5'], test_new[f'{col}_bin5'] = 0, 0
            
            try:
                train_new[f'{col}_bin10'] = pd.qcut(train_new[col], q=10, labels=False, duplicates='drop')
                _, bins10 = pd.qcut(train_new[col], q=10, retbins=True, duplicates='drop')
                test_new[f'{col}_bin10'] = pd.cut(test_new[col], bins=bins10, labels=False, include_lowest=True)
            except: train_new[f'{col}_bin10'], test_new[f'{col}_bin10'] = 0, 0
    
    scaler = StandardScaler()
    train_new[train_new.columns] = scaler.fit_transform(train_new)
    test_new[test_new.columns] = scaler.transform(test_new)
    return train_new, test_new

y_train = train["accident_risk"]
X_train_full, X_test_full = create_frequency_features(train.drop("accident_risk", axis=1), test.copy())
print(f"Features: {X_train_full.shape[1]}")

y_bins = pd.qcut(y_train, q=10, labels=False, duplicates='drop')

# Best parameters from Optuna Trial 40
BEST_PARAMS = {
    'max_depth': 9, 'learning_rate': 0.013341474504337291, 'n_estimators': 1700,
    'subsample': 0.8054543315623267, 'colsample_bytree': 0.807768915571572,
    'min_child_weight': 4, 'gamma': 0.009574146894320856,
    'reg_alpha': 0.0988653659213736, 'reg_lambda': 0.45612897916439415,
    'max_delta_step': 1, 'colsample_bylevel': 0.8572157417230063,
    'colsample_bynode': 0.8838282649012005, 'scale_pos_weight': 0.8272079611457457,
    'max_bin': 512, 'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor',
    'eval_metric': 'rmse', 'random_state': 42
}

print(f"\nExpected RMSE: 0.055947\n")

# Training
FOLDS = 7
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train_full))
test_preds = np.zeros(len(X_test_full))
fold_scores, fold_details, feature_importance_list = [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_bins), 1):
    X_train_fold, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_train_fold, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model = XGBRegressor(**BEST_PARAMS, early_stopping_rounds=50)
    start = time.time()
    model.fit(X_train_fold, y_train_fold, eval_set=[(X_val, y_val)], verbose=0)
    
    val_preds = model.predict(X_val)
    test_preds += model.predict(X_test_full)
    oof_preds[val_idx] = val_preds
    
    rmse = np.sqrt(((val_preds - y_val)**2).mean())
    mae = np.abs(val_preds - y_val).mean()
    r2 = 1 - (np.sum((y_val - val_preds)**2) / np.sum((y_val - y_val.mean())**2))
    
    fold_scores.append(rmse)
    fold_details.append({'fold': fold, 'rmse': rmse, 'mae': mae, 'r2': r2, 
                        'train_time': time.time()-start, 'best_iteration': model.best_iteration})
    feature_importance_list.append(model.feature_importances_)
    print(f"Fold {fold}: RMSE={rmse:.6f} MAE={mae:.6f} R²={r2:.4f}")

test_preds /= FOLDS
cv_rmse, cv_std = np.mean(fold_scores), np.std(fold_scores)

print(f"\nCV RMSE: {cv_rmse:.6f} (±{cv_std:.6f})")

# Save submission
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sub["accident_risk"] = test_preds

# Save main submission file
sub.to_csv("submission.csv", index=False)
print(f"✓ Submission saved: submission.csv")

# Optional: Save backup with timestamp
backup_filename = f"submission_{timestamp}_cv{cv_rmse:.6f}.csv"
sub.to_csv(backup_filename, index=False)
print(f"✓ Backup saved: {backup_filename}")
print(f"  CV RMSE: {cv_rmse:.6f}\n")

# Analysis
residuals = y_train - oof_preds
abs_residuals = np.abs(residuals)
squared_residuals = residuals**2

print("="*80)
print("COMPREHENSIVE ANALYSIS")
print("="*80)

# Statistics
print(f"\nBasic Stats:\n  RMSE: {np.sqrt(squared_residuals.mean()):.6f}")
print(f"  MAE: {abs_residuals.mean():.6f}")
print(f"  Mean residual: {residuals.mean():.6f}")
print(f"  Std residual: {residuals.std():.6f}")

pearson_corr, _ = pearsonr(y_train, oof_preds)
print(f"\nCorrelations:\n  Pearson: {pearson_corr:.6f}")

_, p_val = stats.normaltest(residuals)
print(f"\nNormality: {'Yes' if p_val>0.05 else 'No'} (p={p_val:.6f})")

# Visualizations
fig = plt.figure(figsize=(24, 16))

# 1. Actual vs Predicted
plt.subplot(4, 5, 1)
plt.scatter(y_train, oof_preds, alpha=0.3, s=3)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual'); plt.ylabel('Predicted')
plt.title(f'Actual vs Predicted (r={pearson_corr:.4f})', fontweight='bold')
plt.grid(True, alpha=0.3)

# 2. Residuals vs Predicted
plt.subplot(4, 5, 2)
plt.scatter(oof_preds, residuals, alpha=0.3, s=3)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted'); plt.ylabel('Residuals')
plt.title('Residuals vs Predicted', fontweight='bold')
plt.grid(True, alpha=0.3)

# 3. Residual Distribution
plt.subplot(4, 5, 3)
plt.hist(residuals, bins=100, edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', lw=2)
plt.xlabel('Residuals'); plt.ylabel('Frequency')
plt.title('Residual Distribution', fontweight='bold')
plt.grid(True, alpha=0.3)

# 4. Q-Q Plot
plt.subplot(4, 5, 4)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot', fontweight='bold')
plt.grid(True, alpha=0.3)

# 5. Absolute Residuals
plt.subplot(4, 5, 5)
plt.scatter(oof_preds, abs_residuals, alpha=0.3, s=3)
plt.xlabel('Predicted'); plt.ylabel('Absolute Residuals')
plt.title('Absolute Residuals', fontweight='bold')
plt.grid(True, alpha=0.3)

# 6. Cumulative Error
plt.subplot(4, 5, 6)
sorted_abs = np.sort(abs_residuals)
cumulative = np.arange(1, len(sorted_abs)+1) / len(sorted_abs) * 100
plt.plot(sorted_abs, cumulative, lw=2)
plt.axhline(y=95, color='r', linestyle='--')
plt.xlabel('Absolute Error'); plt.ylabel('Cumulative %')
plt.title('Cumulative Error', fontweight='bold')
plt.grid(True, alpha=0.3)

# 7. RMSE by Target Bins
plt.subplot(4, 5, 7)
target_bins = pd.qcut(y_train, q=20, labels=False, duplicates='drop')
bin_rmse = [np.sqrt(squared_residuals[target_bins==i].mean()) for i in range(target_bins.max()+1)]
plt.bar(range(len(bin_rmse)), bin_rmse, alpha=0.7, edgecolor='black')
plt.xlabel('Target Bins'); plt.ylabel('RMSE')
plt.title('RMSE by Target Bins', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 8. Feature Importance
plt.subplot(4, 5, 8)
avg_imp = np.mean(feature_importance_list, axis=0)
imp_df = pd.DataFrame({'feature': X_train_full.columns, 'importance': avg_imp}).sort_values('importance', ascending=False).head(15)
plt.barh(range(len(imp_df)), imp_df['importance'], alpha=0.7, edgecolor='black')
plt.yticks(range(len(imp_df)), imp_df['feature'], fontsize=7)
plt.xlabel('Importance')
plt.title('Top 15 Features', fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')

# 9. Fold Performance
plt.subplot(4, 5, 9)
plt.bar([f['fold'] for f in fold_details], [f['rmse'] for f in fold_details], alpha=0.7, edgecolor='black')
plt.axhline(y=cv_rmse, color='r', linestyle='--', lw=2, label=f'Mean: {cv_rmse:.6f}')
plt.xlabel('Fold'); plt.ylabel('RMSE')
plt.title('RMSE by Fold', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 10. Residuals Boxplot
plt.subplot(4, 5, 10)
pred_deciles = pd.qcut(oof_preds, q=10, labels=False, duplicates='drop')
res_by_dec = [residuals[pred_deciles==i] for i in range(pred_deciles.max()+1)]
plt.boxplot(res_by_dec, showfliers=False)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Prediction Decile'); plt.ylabel('Residuals')
plt.title('Residuals by Decile', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 11. Training Time
plt.subplot(4, 5, 11)
plt.plot([f['fold'] for f in fold_details], [f['train_time'] for f in fold_details], marker='o', lw=2)
plt.xlabel('Fold'); plt.ylabel('Time (sec)')
plt.title('Training Time', fontweight='bold')
plt.grid(True, alpha=0.3)

# 12. Error Heatmap
plt.subplot(4, 5, 12)
pred_bins_2d = pd.qcut(oof_preds, q=10, labels=False, duplicates='drop')
actual_bins_2d = pd.qcut(y_train, q=10, labels=False, duplicates='drop')
error_matrix = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        mask = (pred_bins_2d==i) & (actual_bins_2d==j)
        if mask.sum() > 0:
            error_matrix[j,i] = np.sqrt(squared_residuals[mask].mean())
sns.heatmap(error_matrix, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'RMSE'})
plt.xlabel('Predicted Bins'); plt.ylabel('Actual Bins')
plt.title('RMSE Heatmap', fontweight='bold')

# 13. Residuals vs Actual
plt.subplot(4, 5, 13)
plt.scatter(y_train, residuals, alpha=0.3, s=3)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Actual'); plt.ylabel('Residuals')
plt.title('Residuals vs Actual', fontweight='bold')
plt.grid(True, alpha=0.3)

# 14. Distribution Comparison
plt.subplot(4, 5, 14)
plt.hist(y_train, bins=50, alpha=0.5, label='Actual', edgecolor='black')
plt.hist(oof_preds, bins=50, alpha=0.5, label='Predicted', edgecolor='black')
plt.xlabel('Value'); plt.ylabel('Frequency')
plt.title('Distribution Comparison', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 15. MAE by Fold
plt.subplot(4, 5, 15)
plt.bar([f['fold'] for f in fold_details], [f['mae'] for f in fold_details], alpha=0.7, edgecolor='black', color='orange')
plt.axhline(y=np.mean([f['mae'] for f in fold_details]), color='r', linestyle='--', lw=2)
plt.xlabel('Fold'); plt.ylabel('MAE')
plt.title('MAE by Fold', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 16. Best Iterations
plt.subplot(4, 5, 16)
plt.bar([f['fold'] for f in fold_details], [f['best_iteration'] for f in fold_details], alpha=0.7, edgecolor='black', color='green')
plt.xlabel('Fold'); plt.ylabel('Best Iteration')
plt.title('Early Stopping Iterations', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 17. R² by Fold
plt.subplot(4, 5, 17)
plt.bar([f['fold'] for f in fold_details], [f['r2'] for f in fold_details], alpha=0.7, edgecolor='black', color='purple')
plt.axhline(y=np.mean([f['r2'] for f in fold_details]), color='r', linestyle='--', lw=2)
plt.xlabel('Fold'); plt.ylabel('R²')
plt.title('R² by Fold', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 18. Error by Quantiles
plt.subplot(4, 5, 18)
quantiles = [(0,25), (25,50), (50,75), (75,100)]
q_rmse = []
for q_low, q_high in quantiles:
    q_low_val, q_high_val = np.percentile(y_train, [q_low, q_high])
    mask = (y_train >= q_low_val) & (y_train <= q_high_val)
    q_rmse.append(np.sqrt(np.mean(squared_residuals[mask])))
plt.bar(range(len(q_rmse)), q_rmse, alpha=0.7, edgecolor='black')
plt.xticks(range(len(q_rmse)), [f'Q{q[0]}-{q[1]}' for q in quantiles])
plt.ylabel('RMSE')
plt.title('RMSE by Target Quantiles', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 19. Absolute Error (Log)
plt.subplot(4, 5, 19)
plt.hist(abs_residuals, bins=100, edgecolor='black', alpha=0.7, log=True)
plt.xlabel('Absolute Error'); plt.ylabel('Log Frequency')
plt.title('Absolute Error (Log Scale)', fontweight='bold')
plt.grid(True, alpha=0.3)

# 20. Prediction Range
plt.subplot(4, 5, 20)
data_range = [y_train.min(), y_train.max(), oof_preds.min(), oof_preds.max()]
labels = ['Actual Min', 'Actual Max', 'Pred Min', 'Pred Max']
colors = ['blue', 'blue', 'orange', 'orange']
plt.bar(range(len(data_range)), data_range, alpha=0.7, edgecolor='black', color=colors)
plt.xticks(range(len(data_range)), labels, rotation=45, ha='right', fontsize=8)
plt.ylabel('Value')
plt.title('Prediction Range Coverage', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"analysis_{timestamp}.png", dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved: analysis_{timestamp}.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

