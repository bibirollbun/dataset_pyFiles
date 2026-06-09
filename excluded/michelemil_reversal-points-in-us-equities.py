import pandas as pd
import numpy as np
import logging
import warnings
import os
from functools import partial

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb

# -------------------------
# CONFIG & LOGGING
# -------------------------
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("reversal_pipeline")

TRAIN_PATH = "/kaggle/input/detecting-reversal-points-in-us-equities/new_comptetition_data/train.csv"
TEST_PATH = "/kaggle/input/detecting-reversal-points-in-us-equities/new_comptetition_data/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/detecting-reversal-points-in-us-equities/new_comptetition_data/sample_submission.csv"
GA_FEATURE_FILE = "/kaggle/input/genetic-algo-jax-feature-selection/final_selected_features.csv"

# -------------------------
# JAX SUBSPACE SOLVER (Kept but not required for LGBM)
# -------------------------
@partial(jit, static_argnums=(4, 5))
def solve_subspace_jit(XtX_full, Xty_full, mask, alpha, num_classes, max_cap):
    sort_idx = jnp.argsort(mask, descending=True)
    active_indices = sort_idx[:max_cap]
    is_active = mask[active_indices]

    XtX_sub = XtX_full[active_indices][:, active_indices]
    Xty_sub = Xty_full[active_indices]

    sub_mask_mat = is_active[:, None] * is_active[None, :]
    XtX_active = XtX_sub * sub_mask_mat

    diag_reg = jnp.where(is_active > 0, alpha, 1.0)
    XtX_reg = XtX_active + jnp.diag(diag_reg) + (jnp.eye(max_cap) * (1.0 - is_active[:, None]))

    Xty_active = Xty_sub * is_active[:, None]
    W_sub = jnp.linalg.solve(XtX_reg, Xty_active)

    return W_sub, active_indices

# -------------------------
# DATA PREPARATION & ROBUST MAPPING
# -------------------------
def load_and_prepare_data():
    logger.info("Loading data...")
    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    test = pd.read_csv(TEST_PATH, low_memory=False)

    # Robust mapping function
    def robust_map(x):
        if pd.isna(x):
            return 'None'
        val = str(x).strip().upper()
        # Map explicit patterns to H or L; keep unknowns as 'None'
        if val in ('H', 'HH', 'LH', 'HIGHLIGHT', 'HIGH'):
            return 'H'
        if val in ('L', 'LL', 'HL', 'LOW'):
            return 'L'
        if val in ('NONE', 'N', ''):
            return 'None'
        # fallback: if contains H or L prefer those
        if 'H' in val and 'L' not in val:
            return 'H'
        if 'L' in val and 'H' not in val:
            return 'L'
        return 'None'

    logger.info("Mapping class labels...")
    y_raw = train['class_label'].apply(robust_map)

    # Diagnostics
    logger.info(f"Unique classes identified: {y_raw.unique().tolist()}")
    logger.info(f"Class distribution:\n{y_raw.value_counts(dropna=False)}")

    if y_raw.nunique() < 2:
        raise ValueError(f"Error: Only one class found: {y_raw.unique().tolist()}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)

    logger.info(f"LabelEncoder classes (order): {list(le.classes_)}")

    # Identify feature columns (excluding metadata)
    meta_cols = ['id', 'train_id', 'Unnamed: 0', 'ticker_id', 't', 'class_label']
    all_numeric_features = [col for col in train.columns if col not in meta_cols]

    # Basic sanity
    logger.info(f"Found {len(all_numeric_features)} candidate features.")

    return train, test, all_numeric_features, y_encoded, le

# Execute preparation
train_df, test_df, all_feats, y_encoded, le = load_and_prepare_data()
n_classes = len(le.classes_)
logger.info(f"n_classes = {n_classes}")

# -------------------------
# FEATURE SELECTION: USE ALL AVAILABLE FEATURES (no GA)
# -------------------------
logger.info("Using all available features (no GA selection).")
selected_features = all_feats.copy()

# Ensure consistent column order and existence in both train/test
selected_features = [f for f in selected_features if f in train_df.columns and f in test_df.columns]
logger.info(f"Using {len(selected_features)} features.")


# Ensure consistent column order and existence
missing_in_train = [f for f in selected_features if f not in train_df.columns]
missing_in_test = [f for f in selected_features if f not in test_df.columns]
if missing_in_train or missing_in_test:
    logger.warning(f"Missing features in train: {missing_in_train}")
    logger.warning(f"Missing features in test: {missing_in_test}")
    # fallback to intersection
    selected_features = [f for f in selected_features if f in train_df.columns and f in test_df.columns]
    logger.info(f"Using {len(selected_features)} features after intersection.")

# Imputation strategy: mean + missingness indicator
def prepare_matrix(df, features):
    X = df[features].copy()
    X_missing = X.isna().astype(np.uint8).add_suffix("_missing")
    X_imputed = X.fillna(X.mean())
    X_final = pd.concat([X_imputed, X_missing], axis=1)
    return X_final

X = prepare_matrix(train_df, selected_features)
X_test = prepare_matrix(test_df, selected_features)

logger.info(f"X shape: {X.shape}, X_test shape: {X_test.shape}")

# Optional scaling (commented out; enable if helpful)
# scaler = StandardScaler()
# X[X.columns] = scaler.fit_transform(X)
# X_test[X_test.columns] = scaler.transform(X_test)

# -------------------------
# LIGHTGBM TRAINING (Stratified 5-Fold) with diagnostics
# -------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((X.shape[0], n_classes))
test_oof_preds = np.zeros((X_test.shape[0], n_classes))

# Compute class weights for balanced learning
cw = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = {int(cls): float(w) for cls, w in zip(np.unique(y_encoded), cw)}
logger.info(f"class_weight_dict: {class_weight_dict}")

compute_fold_f1s = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    logger.info(f"--- Training Fold {fold+1} ---")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y_encoded[train_idx], y_encoded[val_idx]

    # Optional simple upsampling of minority classes in training fold (disabled by default)
    UPSAMPLE_MINORITY = False
    if UPSAMPLE_MINORITY:
        df_tr = pd.concat([X_tr.reset_index(drop=True), pd.Series(y_tr, name='target')], axis=1)
        counts = df_tr['target'].value_counts()
        max_count = counts.max()
        parts = []
        for cls, cnt in counts.items():
            part = df_tr[df_tr['target'] == cls]
            if cnt < max_count:
                part = part.sample(max_count, replace=True, random_state=42)
            parts.append(part)
        df_tr_bal = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
        y_tr = df_tr_bal['target'].values
        X_tr = df_tr_bal.drop(columns=['target'])

    objective_type = 'multiclass' if n_classes > 2 else 'binary'

    # For multiclass, do not pass num_class for binary; let LGBM infer
    model = lgb.LGBMClassifier(
        objective=objective_type,
        num_class=n_classes if n_classes > 2 else None,
        learning_rate=0.03,
        n_estimators=1200,
        class_weight=class_weight_dict,
        random_state=42,
        importance_type='gain',
        verbosity=-1
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss' if n_classes > 2 else 'binary_logloss',
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200)
        ],
    )

    val_proba = model.predict_proba(X_val)
    logger.info(f"val_proba shape: {val_proba.shape}")

    # Ensure shape matches
    if val_proba.ndim == 1:
        # binary case returning single column of proba for positive class
        val_proba = np.vstack([1 - val_proba, val_proba]).T

    oof_preds[val_idx] = val_proba
    test_proba = model.predict_proba(X_test)
    if test_proba.ndim == 1:
        test_proba = np.vstack([1 - test_proba, test_proba]).T
    test_oof_preds += test_proba / skf.n_splits

    val_preds = np.argmax(val_proba, axis=1)
    fold_f1 = f1_score(y_val, val_preds, average='macro')
    compute_fold_f1s.append(fold_f1)
    logger.info(f"Fold {fold+1} macro-F1: {fold_f1:.6f}")

# Overall OOF F1
oof_preds_labels = np.argmax(oof_preds, axis=1)
oof_macro_f1 = f1_score(y_encoded, oof_preds_labels, average='macro')
logger.info(f"OOF macro-F1: {oof_macro_f1:.6f}")
logger.info(f"Per-fold macro-F1s: {compute_fold_f1s}")

# -------------------------
# SUBMISSION & MANUAL OVERRIDES (by sample ID)
# -------------------------
final_labels = le.inverse_transform(np.argmax(test_oof_preds, axis=1))

submission = pd.read_csv(SAMPLE_SUB_PATH)

# Validate sample submission has expected id column
if 'id' not in submission.columns:
    logger.warning("sample_submission has no 'id' column; manual overrides by id won't be applied.")
submission['class_label'] = final_labels

# If you need manual overrides, use sample IDs here (example keys must be sample ids, not indices)
# Example: overrides_by_id = {'sample_0001': 'L', 'sample_0758': 'H'}
overrides_by_id = {}  # populate with actual sample ids if needed
if overrides_by_id:
    for sample_id, lbl in overrides_by_id.items():
        submission.loc[submission['id'] == sample_id, 'class_label'] = lbl
    logger.info(f"Applied {len(overrides_by_id)} manual overrides by sample id.")

submission.to_csv("submission.csv", index=False)
logger.info("Pipeline complete. 'submission.csv' generated.")



submission.head(5)


submission.shape, submission.isna().sum()

