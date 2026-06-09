import numpy as np
import pandas as pd
import torch
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150


# Paths
TRAIN_PATH = '/kaggle/input/playground-series-s5e12/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e12/test.csv'
ORIG_PATH = '/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv'

# Constants
TARGET = 'diagnosed_diabetes'
ID_COL = 'id'
RANDOM_STATE = 42

# Strategy parameters
TAIL_WEIGHT = 5.0
CUTOFF_ID = 678260
N_SPLITS = 10
SEEDS = [42, 123, 777]
BIN_Q = 20

# Hardware check
task_device = 'GPU' if torch.cuda.is_available() else 'CPU'
print(f"ğŸ�¬ Environment Check: Device set to >> {task_device} <<")


def fit_stat_binning(series, q=20):
    """Learn quantile bin boundaries from the training set."""
    _, bins = pd.qcut(series.dropna().rank(method='first'), q=q, retbins=True, duplicates='drop')
    return bins

def apply_stat_binning(series, bins):
    """Apply learned boundaries to any dataset."""
    return pd.cut(series, bins=bins, labels=False, include_lowest=True).astype(str)

def fit_ai_binning(X, y, col, max_depth=3):
    """Learn decision tree thresholds from the training set."""
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    X_filled = X[[col]].fillna(X[col].mean())
    dt.fit(X_filled, y)
    return sorted(list(set([t for t in dt.tree_.threshold if t != -2])))

def apply_ai_binning(series, thresholds):
    """Apply learned AI thresholds to any dataset."""
    bins = [-np.inf] + thresholds + [np.inf]
    return pd.cut(series, bins=bins, labels=False).astype(str)

def add_orig_stats(orig_df, df_to_update, numeric_cols):
    """Map diabetes risk statistics from the original dataset."""
    for col in numeric_cols:
        if col in [ID_COL, TARGET] or col not in orig_df.columns: continue
        
        mean_map = orig_df.groupby(col)[TARGET].mean()
        count_map = orig_df.groupby(col).size()
        global_mean = orig_df[TARGET].mean()
        
        df_to_update[f'orig_mean_{col}'] = df_to_update[col].map(mean_map).fillna(global_mean)
        df_to_update[f'orig_count_{col}'] = df_to_update[col].map(count_map).fillna(0).astype(int)
    return df_to_update


print("ğŸ“‚ Loading data...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
orig = pd.read_csv(ORIG_PATH)

binning_targets = [
    'age', 'bmi', 'triglycerides', 'physical_activity_minutes_per_week', 
    'cholesterol_total', 'ldl_cholesterol', 'hdl_cholesterol', 
    'systolic_bp', 'diastolic_bp', 'sleep_hours_per_day', 'diet_score'
]

print("ğŸ”§ Note: Binning (Stat/AI) will be fitted inside each CV fold to prevent target leakage.")

# Original Stats mapping (kept global, no leakage here)
num_cols = train.select_dtypes(include=[np.number]).columns
train = add_orig_stats(orig, train, num_cols)
test = add_orig_stats(orig, test, num_cols)

print(f"âœ… Base features prepared (bin_* features will be added per-fold during CV)")


# ğŸš€ Training Loop (Seeds x Folds) â€” Fold-wise Binning to avoid target leakage
# Model Hyperparameters
params = {
    'learning_rate': 0.089,
    'depth': 6,
    'l2_leaf_reg': 15.7,
    'bagging_temperature': 0.898,
    'random_strength': 0.0006,
    'iterations': 2000,
    'early_stopping_rounds': 50,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'task_type': task_device,
    'logging_level': 'Silent',
    'metric_period': 100,
    'allow_writing_files': False
}

# Weighting setup
weights = np.ones(len(train))
weights[train[ID_COL] >= CUTOFF_ID] = TAIL_WEIGHT

print("ğŸš€ Starting Training with Fold-wise Binning...")

oof_matrix = np.zeros((len(SEEDS), len(train)))
test_matrix = np.zeros((len(SEEDS), len(test)))

for s, seed in enumerate(SEEDS):
    print(f"\nğŸŒ± Seed {seed} ({s+1}/{len(SEEDS)})")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    seed_test_preds = np.zeros(len(test))

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train, train[TARGET]), 1):
        # 1) Split raw data (no binning yet)
        X_tr = train.iloc[trn_idx].copy()
        y_tr = train[TARGET].iloc[trn_idx]
        w_tr = weights[trn_idx]

        X_val = train.iloc[val_idx].copy()
        y_val = train[TARGET].iloc[val_idx]

        # Test copy for fold-specific transformations
        X_test_fold = test.copy()

        # 2) Fit & apply binning on training fold only
        # A. Statistical binning (qcut)
        for col in binning_targets:
            s_bins = fit_stat_binning(X_tr[col], q=BIN_Q)
            X_tr[f'bin_{col}_stat'] = apply_stat_binning(X_tr[col], s_bins)
            X_val[f'bin_{col}_stat'] = apply_stat_binning(X_val[col], s_bins)
            X_test_fold[f'bin_{col}_stat'] = apply_stat_binning(X_test_fold[col], s_bins)

        # B. AI Binning (DecisionTree thresholds) â€” fitted only on X_tr and y_tr
        for col in binning_targets:
            a_thresholds = fit_ai_binning(X_tr, y_tr, col)
            X_tr[f'bin_{col}_ai'] = apply_ai_binning(X_tr[col], a_thresholds)
            X_val[f'bin_{col}_ai'] = apply_ai_binning(X_val[col], a_thresholds)
            X_test_fold[f'bin_{col}_ai'] = apply_ai_binning(X_test_fold[col], a_thresholds)

        # 3) Prepare fold features and categorical list
        fold_features = [c for c in X_tr.columns if c not in [ID_COL, TARGET]]
        fold_cat_features = [c for c in fold_features if (X_tr[c].dtype == 'object') or ('bin_' in c)]

        # 4) Train model on this fold
        train_pool = Pool(X_tr[fold_features], y_tr, cat_features=fold_cat_features, weight=w_tr)
        val_pool = Pool(X_val[fold_features], y_val, cat_features=fold_cat_features)

        model = CatBoostClassifier(**params, random_seed=seed)
        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            early_stopping_rounds=params["early_stopping_rounds"],
            verbose=False
        )

        # 5) Predict
        val_pred = model.predict_proba(X_val[fold_features])[:, 1]
        oof_matrix[s, val_idx] = val_pred

        seed_test_preds += model.predict_proba(X_test_fold[fold_features])[:, 1] / N_SPLITS

        # store last fold features for later analysis/importance plots
        last_fold_features = fold_features

    test_matrix[s] = seed_test_preds
    print(f"  ğŸ�† Seed {seed} OOF AUC: {roc_auc_score(train[TARGET], oof_matrix[s]):.5f}")

# Average across seeds and report
final_oof = oof_matrix.mean(axis=0)
final_test = test_matrix.mean(axis=0)
print(f"\nğŸ�† FINAL SEED-AVERAGED OOF AUC: {roc_auc_score(train[TARGET], final_oof):.6f}")


# --- Visualization: Feature Importance ---
print("\nğŸ“Š Generating Feature Importance Plots...")

# Get all feature importances (from last trained model)
feat_imp = model.get_feature_importance()
imp_df_full = pd.DataFrame({'feature': last_fold_features, 'importance': feat_imp})

# 1. Overall Top 20
top20_df = imp_df_full.sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=top20_df, palette='viridis')
plt.title("Overall Top 20 Important Features", fontsize=14, fontweight='bold')
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 2. Binning Features Only Top 10
bin_df = imp_df_full[imp_df_full['feature'].str.contains('bin_')].sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x='importance', y='feature', data=bin_df, palette='magma')
plt.title("Top 10 'Binning' Features Only", fontsize=14, fontweight='bold')
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


# Save OOF with Target for easy ensembling
pd.DataFrame({
    ID_COL: train[ID_COL], 
    TARGET: train[TARGET], 
    'oof_pred': final_oof
}).to_csv('catboost_final_oof.csv', index=False)

# Save Submission
pd.DataFrame({
    ID_COL: test[ID_COL], 
    TARGET: final_test
}).to_csv('submission.csv', index=False)

print("ğŸ’¾ Files 'catboost_final_oof.csv' and 'submission.csv' saved.")

# Distribution Check
plt.figure(figsize=(10, 5))
sns.kdeplot(final_oof, label='Train OOF', fill=True, color='blue', alpha=0.3)
sns.kdeplot(final_test, label='Test Predictions', fill=True, color='orange', alpha=0.3)
plt.title("Final Prediction Distribution Consistency")
plt.legend()
plt.show()

