import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import lightgbm as lgb
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

print("="*80)
print("LGBM + ORIG FEATURES + 20 SEEDS")
print("="*80)



print("\n SECTION 1: Loading Data...")

# Load competition data
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

# Load original datasets (external knowledge base)
orig_100k = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')
orig_10k = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_10k.csv')
orig_2k = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_2k.csv')

# Concatenate all original data
orig = pd.concat([orig_100k, orig_10k, orig_2k], axis=0, ignore_index=True)

print(f"Train Shape: {train.shape}")
print(f"Test Shape: {test.shape}")
print(f"Original Data Shape: {orig.shape}")
print(f"   Original = {orig_100k.shape[0]} + {orig_10k.shape[0]} + {orig_2k.shape[0]} = {orig.shape[0]} rows")





print("\n SECTION 2: Defining Feature Categories...")

TARGET = 'accident_risk'

# Base features (all columns except id and target)
BASE = [col for col in train.columns if col not in ['id', TARGET]]

# Categorical features (for LightGBM categorical handling)
CATS = ['road_type', 'lighting', 'weather', 'road_signs_present', 
        'public_road', 'time_of_day', 'holiday', 'school_season']

print(f"\n {len(BASE)} Base Features:")
for i, feat in enumerate(BASE, 1):
    print(f"   {i:2d}. {feat}")


print("\n SECTION 3: Creating ORIG Features (External Target Encoding)...")
print("   Strategy: Use original dataset statistics as features")

ORIG = []

for col in BASE:
    # Calculate mean accident_risk for each unique value in original data
    orig_stats = orig.groupby(col)[TARGET].mean()
    
    new_col_name = f"orig_{col}"
    
    # Map these statistics to train and test
    train[new_col_name] = train[col].map(orig_stats)
    test[new_col_name] = test[col].map(orig_stats)
    
    ORIG.append(new_col_name)
    
    # Handle any missing values (shouldn't happen, but safety)
    if train[new_col_name].isnull().any():
        # Fill with overall mean from original data
        overall_mean = orig[TARGET].mean()
        train[new_col_name].fillna(overall_mean, inplace=True)
        test[new_col_name].fillna(overall_mean, inplace=True)

print(f" Created {len(ORIG)} ORIG features (external target encoding)")

# Show example of what ORIG features capture
print("\n Example: orig_speed_limit statistics")
speed_stats = orig.groupby('speed_limit')[TARGET].agg(['mean', 'std', 'count'])
print(speed_stats)


print("\n SECTION 4: Creating META Feature (Formula)...")

META = []

for df in [train, test, orig]:
    base_risk = (
        0.3 * df["curvature"] + 
        0.2 * (df["lighting"] == "night").astype(int) + 
        0.1 * (df["weather"] != "clear").astype(int) + 
        0.2 * (df["speed_limit"] >= 60).astype(int) + 
        0.1 * (np.array(df["num_reported_accidents"]) > 2).astype(int)
    )
    df['Meta'] = base_risk

META.append('Meta')

print(" Created META feature (formula as feature)")
print(f"   Formula: 0.3*curve + 0.2*night + 0.1*bad_weather + 0.2*high_speed + 0.1*many_accidents")


print("\n SECTION 5: Preparing Final Feature Set...")

FEATURES = BASE + ORIG + META

print(f"\n Total Features: {len(FEATURES)}")
print(f"   - BASE: {len(BASE)} features")
print(f"   - ORIG: {len(ORIG)} features (external target encoding)")
print(f"   - META: {len(META)} feature (formula)")

# Prepare X and y
X = train[FEATURES].copy()
y = train[TARGET].values
print(f"\n Feature Matrix Shape: {X.shape}")
print(f" Target Shape: {y.shape}")


print("\n  SECTION 6: LightGBM Parameters...")

params = {
    'num_leaves': 16,
    'max_depth': 10,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'reg_alpha': 1, 
    'reg_lambda': 0.1, 
    'max_bin': 255,
    'n_estimators': 100000,
    'learning_rate': 0.01,
    'device': 'gpu',
    'verbosity': -1,
    'random_state': None  # Will be set per seed
}

print("Parameters:")
for key, value in params.items():
    if key != 'random_state':
        print(f"   {key}: {value}")


print("\n SECTION 7: Cross-Validation Setup...")

N_SPLITS = 7
RANDOM_SEED = 42

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

print(f" Using {N_SPLITS}-Fold Cross-Validation")
print(f" CV Random Seed: {RANDOM_SEED}")


print("\n" + "="*80)
print(" SECTION 8: Training LGBM with 20 Random Seeds")
print("="*80)

# Generate 20 random seeds
population = range(1, 100001)
SEEDS = random.sample(population, 20)

print(f"\n Generated 20 Random Seeds:")
print(f"   {SEEDS}")

# Initialize prediction arrays
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

# Track fold scores
fold_scores = []

print("\n" + "="*80)
print("Starting Training...")
print("="*80)

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"\n{'='*80}")
    print(f" FOLD {fold}/{N_SPLITS}")
    print(f"{'='*80}")
    
    # Split data
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y[train_idx], y[val_idx]
    
    X_test = test[FEATURES].copy()
    
    # Convert categoricals
    X_train[CATS] = X_train[CATS].astype('category')    
    X_val[CATS] = X_val[CATS].astype('category')    
    X_test[CATS] = X_test[CATS].astype('category')
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    
    # Train with 20 different seeds
    fold_oof = np.zeros(len(val_idx))
    fold_test = np.zeros(len(test))
    
    for seed_idx, seed in enumerate(SEEDS, 1):
        print(f"\n    Seed {seed_idx}/20 (seed={seed})...", end=" ")
        
        # Create model with this seed
        model = lgb.LGBMRegressor(**{**params, 'random_state': seed})
        
        # Train
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=500, verbose=False),
                lgb.log_evaluation(0)  # Silent
            ]
        )
        
        # Predict
        fold_oof += model.predict(X_val) / len(SEEDS)
        fold_test += model.predict(X_test) / len(SEEDS)
        
        print(f"âœ“ (iterations: {model.best_iteration_})")
    
    # Store predictions for this fold
    oof_preds[val_idx] = fold_oof
    test_preds += fold_test / N_SPLITS
    
    # Calculate fold score
    fold_rmse = np.sqrt(mean_squared_error(y_val, fold_oof))
    fold_scores.append(fold_rmse)
    
    print(f"\n{'â”€'*80}")
    print(f" Fold {fold} RMSE: {fold_rmse:.6f}")
    print(f"{'â”€'*80}")


# print("\n" + "="*80)
# print(" SECTION 9: Overall Results")
# print("="*80)

# overall_oof = np.sqrt(mean_squared_error(y, oof_preds))
# cv_std = np.std(fold_scores)

# print(f"\n CROSS-VALIDATION RESULTS:")
# print("â”€" * 80)
# for fold_num, score in enumerate(fold_scores, 1):
#     print(f"   Fold {fold_num}: {score:.6f}")
# print("â”€" * 80)
# print(f"   Mean:   {np.mean(fold_scores):.6f}")
# print(f"   Std:    {cv_std:.6f}")
# print(f"\n OVERALL OOF RMSE: {overall_oof:.5f}")
# print("â”€" * 80)

# # Compare with target
# target_lb = 0.05537
# expected_lb = overall_oof - 0.0005  # Historical CV-LB gap

# print(f"\n LEADERBOARD PROJECTION:")
# print(f"   Current OOF:     {overall_oof:.5f}")
# print(f"   Expected LB:     ~{expected_lb:.5f}")
# print(f"   Target LB:       {target_lb:.5f}")
# print(f"   Gap:             {expected_lb - target_lb:+.5f}")

# if expected_lb < target_lb:
#     print(f"\n   âœ… SHOULD BEAT CURRENT LEADER! ğŸ�†")
# else:
#     gap_needed = expected_lb - target_lb
#     print(f"\n     Need {gap_needed:.5f} more improvement")
#     print(f"   ğŸ’¡ Next steps: Add TabM, RealMLP, NN stacking")



# # ============================================================================
# # SECTION 10: FEATURE IMPORTANCE ANALYSIS
# # ============================================================================
# print("\n" + "="*80)
# print(" SECTION 10: Feature Importance Analysis")
# print("="*80)

# # Get feature importance from last model
# feature_importance = model.feature_importances_

# importance_df = pd.DataFrame({
#     'feature': FEATURES, 
#     'importance': feature_importance
# }).sort_values('importance', ascending=False)

# print("\n TOP 30 FEATURES:")
# print("â”€" * 80)
# for idx, row in importance_df.head(30).iterrows():
#     feat_type = "BASE" if row['feature'] in BASE else ("ORIG" if row['feature'].startswith('orig_') else "META")
#     print(f"   {row['feature']:<35} {row['importance']:>10.1f}  [{feat_type}]")

# # Analyze feature type distribution
# orig_count = importance_df.head(20)['feature'].str.startswith('orig_').sum()
# base_count = importance_df.head(20)['feature'].isin(BASE).sum()
# meta_count = importance_df.head(20)['feature'].isin(META).sum()

# print(f"\n TOP 20 FEATURE BREAKDOWN:")
# print(f"   ORIG features: {orig_count}/20 ({orig_count/20*100:.1f}%)")
# print(f"   BASE features: {base_count}/20 ({base_count/20*100:.1f}%)")
# print(f"   META features: {meta_count}/20 ({meta_count/20*100:.1f}%)")


# ============================================================================
# SECTION 11: VISUALIZATIONS
# ============================================================================
print("\n SECTION 11: Creating Visualizations...")

# Plot 1: Feature Importance
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(12, 16))

# Color by feature type
colors = []
for feat in importance_df.head(30)['feature']:
    if feat.startswith('orig_'):
        colors.append('#e74c3c')  # Red for ORIG
    elif feat in META:
        colors.append('#2ecc71')  # Green for META
    else:
        colors.append('#3498db')  # Blue for BASE

ax.barh(range(len(importance_df.head(30))), 
        importance_df.head(30)['importance'].values,
        color=colors)
ax.set_yticks(range(len(importance_df.head(30))))
ax.set_yticklabels(importance_df.head(30)['feature'].values, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Feature Importance - Top 30 Features\n(Red=ORIG, Blue=BASE, Green=META)', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', label='ORIG (external encoding)'),
    Patch(facecolor='#3498db', label='BASE (raw features)'),
    Patch(facecolor='#2ecc71', label='META (formula)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('feature_importance_lgbm_20seeds.png', dpi=300, bbox_inches='tight')
print(" Saved: feature_importance_lgbm_20seeds.png")
plt.show()

# Plot 2: OOF vs True Values
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(y, oof_preds, alpha=0.3, s=1)
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('True Values', fontsize=12)
ax.set_ylabel('OOF Predictions', fontsize=12)
ax.set_title(f'OOF Predictions vs True Values\nRMSE: {overall_oof:.5f}', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('oof_vs_true_lgbm_20seeds.png', dpi=300, bbox_inches='tight')
print(" Saved: oof_vs_true_lgbm_20seeds.png")
plt.show()


# ============================================================================
# SECTION 12: SAVE PREDICTIONS
# ============================================================================
print("\n SECTION 12: Saving Predictions...")

# Clip predictions to valid range
oof_preds = np.clip(oof_preds, 0, 1)
test_preds = np.clip(test_preds, 0, 1)

# Save OOF predictions
oof_df = pd.DataFrame({
    'id': train['id'], 
    TARGET: oof_preds
})
oof_df.to_csv('oof_lgbm_20seeds_origcol.csv', index=False)
print(f" Saved OOF: oof_lgbm_20seeds_origcol.csv")

# Save test predictions (submission)
submission = pd.DataFrame({
    'id': test['id'], 
    TARGET: test_preds
})
submission.to_csv('submission_lgbm_20seeds_origcol.csv', index=False)
print(f" Saved Submission: submission_lgbm_20seeds_origcol.csv")

print(f"\n Submission Statistics:")
print(f"   Min:  {test_preds.min():.4f}")
print(f"   Max:  {test_preds.max():.4f}")
print(f"   Mean: {test_preds.mean():.4f}")
print(f"   Std:  {test_preds.std():.4f}")


# ============================================================================
# SECTION 13: FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print(" TRAINING COMPLETE!")
print("="*80)

print(f"\n SUMMARY:")
print(f"   Strategy: LGBM + ORIG features + 20 seeds")
print(f"   Features: {len(FEATURES)} ({len(BASE)} BASE + {len(ORIG)} ORIG + {len(META)} META)")
print(f"   Models Trained: {N_SPLITS} folds Ã— {len(SEEDS)} seeds = {N_SPLITS * len(SEEDS)} models")
print(f"   OOF RMSE: {overall_oof:.5f}")




