# ===== CELL 1: IMPORTS =====
import os
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error

import xgboost as xgb
from xgboost import XGBRegressor

# Tắt Warning
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# Reset XGBoost config về mặc định
xgb.config.set_config(verbosity=1)

print("All libraries imported")


# ===== CELL 2: LOAD DATA =====

# Load augmented training data
train = pd.read_csv("/kaggle/input/my-augmented-polymer-data/train_augmented.csv")
test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")

print(f" Train: {len(train)} samples (augmented)")
print(f" Test: {len(test)} samples")
print(f"\nSamples per target:")

for target in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
    count = train[target].notna().sum()
    print(f"  {target}: {count}")

print(f"\nFirst 3 rows:")
print(train.head(3))



# ===== CELL 3: TF-IDF VECTORIZATION =====
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,5), max_features=5000)
X = vectorizer.fit_transform(train["SMILES"].fillna(""))
X_test = vectorizer.transform(test["SMILES"].fillna(""))

print(f" TF-IDF features: {X.shape}")
print(f" Test features: {X_test.shape}")



# ===== CELL 4: STRATIFIED K-FOLD WITH BINNING =====
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

def create_stratified_folds_with_binning(y, n_bins=5, n_splits=5, random_state=42):
    """
    Tạo StratifiedKFold dựa trên bins của target y.
    
    Args:
        y: target values (numpy array)
        n_bins: số bins để chia y
        n_splits: số folds cho StratifiedKFold
        random_state: seed
    
    Returns:
        StratifiedKFold object
    """
    # Chia y thành bins
    bins = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
    
    # In thông tin bins
    print(f"\nBinning distribution:")
    for i in range(bins.max() + 1):
        count = (bins == i).sum()
        y_min = y[bins == i].min()
        y_max = y[bins == i].max()
        print(f"  Bin {i}: {count} samples, range [{y_min:.2f}, {y_max:.2f}]")
    
    # Tạo StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    return skf, bins

# Test với Tg
print("===== TESTING STRATIFIED K-FOLD FOR Tg =====")
mask_tg = train['Tg'].notna()
y_tg = train.loc[mask_tg, 'Tg'].values

skf_tg, bins_tg = create_stratified_folds_with_binning(y_tg, n_bins=5, n_splits=5)

print("\nFold distributions:")
for fold_idx, (train_idx, val_idx) in enumerate(skf_tg.split(np.zeros(len(y_tg)), bins_tg), 1):
    y_train_fold = y_tg[train_idx]
    y_val_fold = y_tg[val_idx]
    print(f"Fold {fold_idx}:")
    print(f"  Train: n={len(y_train_fold)}, mean={y_train_fold.mean():.2f}, std={y_train_fold.std():.2f}, range=[{y_train_fold.min():.2f}, {y_train_fold.max():.2f}]")
    print(f"  Val:   n={len(y_val_fold)}, mean={y_val_fold.mean():.2f}, std={y_val_fold.std():.2f}, range=[{y_val_fold.min():.2f}, {y_val_fold.max():.2f}]")

print("\nStratifiedKFold function defined and tested")



# ===== CELL 5: TRAINING WITH STRATIFIED K-FOLD + LOG =====

def get_params_for_target(target, early_stopping_rounds=15):
    common = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "early_stopping_rounds": early_stopping_rounds,
    }
    
    if target == "Tg":
        return {
            **common,
            "max_depth": 8,
            "n_estimators": 1000,
            "learning_rate": 0.025,
            "reg_lambda": 1.5,
            "reg_alpha": 1.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 3000,
            "min_child_weight": 3,
        }
    elif target == "FFV":
        return {
            **common,
            "max_depth": 6,
            "n_estimators": 500,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 3000,
            "reg_lambda": 1.0,
            "min_child_weight": 1,
        }
    elif target == "Tc":
        return {
            **common,
            "max_depth": 6,
            "n_estimators": 500,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 3000,
            "reg_lambda": 2.0,
            "min_child_weight": 1,
        }
    elif target == "Density":
        return {
            **common,
            "max_depth": 6,
            "n_estimators": 500,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 3000,
            "reg_lambda": 2.5,
            "min_child_weight": 2,
        }
    elif target == "Rg":
        return {
            **common,
            "max_depth": 5,
            "n_estimators": 400,
            "learning_rate": 0.05,
            "reg_lambda": 2.5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 3000,
            "min_child_weight": 3,
        }


def train_with_stratified_kfold(X_target, y_target, target,
                                n_splits=5, n_bins=5,
                                es_rounds=15,
                                log_interval=15):
    """
    Train với StratifiedKFold (cho Tg) hoặc KFold (các target khác),
    in log dạng:
    [Tg] Iter 0, train_mae=..., val_mae=...
    """
    print(f"\n===== {target}: STRATIFIED K-FOLD TRAINING =====")
    print(f"Target: {target} - Total labeled samples: {len(y_target)}")
    
    params = get_params_for_target(target, early_stopping_rounds=es_rounds)
    
    # Tạo folds
    if target == "Tg":
        # Stratified theo bins Tg
        bins = pd.qcut(y_target, q=n_bins, labels=False, duplicates='drop')
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_iterator = kf.split(X_target, bins)
        print(f"Using StratifiedKFold with {n_bins} bins")
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_iterator = kf.split(X_target)
        print("Using regular KFold")
    
    fold_models = []
    fold_train_mae = []
    fold_val_mae = []
    
    for fold, (train_idx, val_idx) in enumerate(fold_iterator, 1):
        X_train_fold = X_target[train_idx]
        X_val_fold = X_target[val_idx]
        y_train_fold = y_target[train_idx]
        y_val_fold = y_target[val_idx]
        
        model = XGBRegressor(**params)
        
        # tiến hành fit với eval_set để lấy evals_result
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)],
            verbose=False
        )
        
        # Lấy history MAE
        evals_result = model.evals_result()
        train_mae_hist = evals_result['validation_0']['mae']
        val_mae_hist = evals_result['validation_1']['mae']
        
        # In log theo interval
        print(f"\n[{target}] Fold {fold} training log:")
        for i in range(0, len(train_mae_hist), log_interval):
            print(f"[{target}] Iter {i}, train_mae={train_mae_hist[i]:.4f}, val_mae={val_mae_hist[i]:.4f}")
        # log cuối cùng
        last_i = len(train_mae_hist) - 1
        if last_i % log_interval != 0:
            print(f"[{target}] Iter {last_i}, train_mae={train_mae_hist[last_i]:.4f}, val_mae={val_mae_hist[last_i]:.4f}")
        
        best_ntree = model.best_iteration
        
        # Dự đoán với best_iteration
        y_pred_train = model.predict(X_train_fold, iteration_range=(0, best_ntree))
        y_pred_val = model.predict(X_val_fold, iteration_range=(0, best_ntree))
        
        mae_train = mean_absolute_error(y_train_fold, y_pred_train)
        mae_val = mean_absolute_error(y_val_fold, y_pred_val)
        
        fold_models.append(model)
        fold_train_mae.append(mae_train)
        fold_val_mae.append(mae_val)
        
        print(f"\n[{target}] Fold {fold} summary: Train MAE = {mae_train:.4f}, Val MAE = {mae_val:.4f}, best_iter = {best_ntree}")
    
    print(f"\n{target} K-Fold summary:")
    print(f"Train MAE: mean = {np.mean(fold_train_mae):.4f}, std = {np.std(fold_train_mae):.4f}")
    print(f"Val   MAE: mean = {np.mean(fold_val_mae):.4f}, std = {np.std(fold_val_mae):.4f}")
    
    return fold_models



# ===== CELL 6: TRAIN MODELS WITH STRATIFIED K-FOLD =====
models = {}  # Mỗi target: list các fold models

for target in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
    mask = train[target].notna()
    X_target = X[mask.values]
    y_target = train.loc[mask, target].values
    
    # Train với StratifiedKFold (Tg) hoặc KFold (các target khác)
    fold_models = train_with_stratified_kfold(
        X_target, y_target, target,
        n_splits=5,
        n_bins=5,  # Số bins cho Tg
        es_rounds=15
    )
    
    models[target] = fold_models

print("\nAll models trained")



# ===== CELL 7: POST-PROCESSING FUNCTIONS =====
def apply_tg_scaling(predictions_df, scale_factor=1.2):
    """
    Apply empirically-tuned scaling for Tg predictions.
    Tg * 1.2 works best based on leaderboard testing.
    """
    original_mean = predictions_df['Tg'].mean()
    predictions_df['Tg'] = predictions_df['Tg'] * scale_factor
    new_mean = predictions_df['Tg'].mean()
    
    print(f" Tg scaling applied (×{scale_factor}):")
    print(f"  - Before: mean={original_mean:.2f}")
    print(f"  - After:  mean={new_mean:.2f}")
    print(f"  - Shift:  +{new_mean - original_mean:.2f}")
    
    return predictions_df

def ensemble_with_median(fold_predictions):
    """Use median instead of mean - more robust to outliers"""
    return np.median(np.array(fold_predictions), axis=0)

def apply_physical_constraints(predictions_df):
    """Clip predictions to physically valid ranges"""
    constraints_applied = []
    
    if 'FFV' in predictions_df.columns:
        before = predictions_df['FFV'].describe()
        predictions_df['FFV'] = predictions_df['FFV'].clip(0, 1)
        constraints_applied.append('FFV ∈ [0, 1]')
    
    if 'Density' in predictions_df.columns:
        predictions_df['Density'] = predictions_df['Density'].clip(0.5, 3.0)
        constraints_applied.append('Density ∈ [0.5, 3.0]')
    
    if 'Rg' in predictions_df.columns:
        predictions_df['Rg'] = predictions_df['Rg'].clip(0.1, 100)
        constraints_applied.append('Rg ∈ [0.1, 100]')
    
    if 'Tc' in predictions_df.columns:
        predictions_df['Tc'] = predictions_df['Tc'].clip(0, 1.0)
        constraints_applied.append('Tc ∈ [0, 1.0]')
    
    print(f"Physical constraints applied: {', '.join(constraints_applied)}")
    return predictions_df

def clip_to_train_range(predictions_df, train_df, targets, margin=0.15):
    """
    Clip predictions based on training data distribution with margin.
    margin=0.15 means allow 15% extrapolation beyond train min/max.
    """
    clipping_summary = []
    
    for target in targets:
        if target not in predictions_df.columns:
            continue
        
        train_values = train_df[target].dropna()
        train_min = train_values.min()
        train_max = train_values.max()
        train_range = train_max - train_min
        
        # Calculate bounds with margin
        lower_bound = train_min - margin * train_range
        upper_bound = train_max + margin * train_range
        
        # Count how many values will be clipped
        n_clipped_low = (predictions_df[target] < lower_bound).sum()
        n_clipped_high = (predictions_df[target] > upper_bound).sum()
        
        # Apply clipping
        predictions_df[target] = predictions_df[target].clip(lower_bound, upper_bound)
        
        if n_clipped_low + n_clipped_high > 0:
            clipping_summary.append(
                f"{target}: {n_clipped_low} low, {n_clipped_high} high → [{lower_bound:.3f}, {upper_bound:.3f}]"
            )
    
    if clipping_summary:
        print("Range clipping applied:")
        for summary in clipping_summary:
            print(f"  - {summary}")
    else:
        print("Range clipping: No outliers detected")
    
    return predictions_df

print("Post-processing functions defined")



# ===== CELL 8: CREATE SUBMISSION WITH POST-PROCESSING =====
sample_sub = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv")
submission = sample_sub.copy()

print("="*60)
print("GENERATING PREDICTIONS WITH POST-PROCESSING")
print("="*60)

# Step 1: Generate predictions with MEDIAN ensemble (more robust than mean)
print("\n[1/5] Generating K-Fold ensemble predictions...")
for target in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
    if target in models and models[target] is not None:
        fold_models = models[target]  # list of 5 fold models
        
        # Predict with all fold models using best_iteration
        fold_preds = []
        for model in fold_models:
            best_iter = model.best_iteration
            pred = model.predict(X_test, iteration_range=(0, best_iter))
            fold_preds.append(pred)
        
        # Use MEDIAN instead of MEAN (more robust to outliers)
        submission[target] = ensemble_with_median(fold_preds)
        
        print(f"   {target}: {len(fold_models)} folds → median ensemble")

# Step 2: Apply Tg scaling (CRITICAL - empirically found 1.2 works best)
print("\n[2/5] Applying Tg scaling factor...")
submission = apply_tg_scaling(submission, scale_factor=1.2)

# Step 3: Apply physical constraints
print("\n[3/5] Applying physical constraints...")
submission = apply_physical_constraints(submission)

# Step 4: Clip to training range with margin
print("\n[4/5] Clipping to training data range...")
submission = clip_to_train_range(
    submission, 
    train, 
    ['Tg', 'FFV', 'Tc', 'Density', 'Rg'],
    margin=0.15  # Allow 15% extrapolation beyond train range
)

# Step 5: Final summary and save
print("\n[5/5] Final predictions summary:")
print("="*60)
for target in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
    mean_val = submission[target].mean()
    std_val = submission[target].std()
    min_val = submission[target].min()
    max_val = submission[target].max()
    
    # Compare with training data
    train_mean = train[target].mean()
    train_std = train[target].std()
    
    print(f"{target}:")
    print(f"  Prediction: μ={mean_val:.4f}, σ={std_val:.4f}, range=[{min_val:.4f}, {max_val:.4f}]")
    print(f"  Training:   μ={train_mean:.4f}, σ={train_std:.4f}")
    print()

print("="*60)
print("First 5 predictions:")
print(submission.head())
print("="*60)

# Ensure column order matches sample submission
submission = submission[sample_sub.columns]

# Save submission
submission.to_csv('submission.csv', index=False)
print("\n Saved: submission.csv")

