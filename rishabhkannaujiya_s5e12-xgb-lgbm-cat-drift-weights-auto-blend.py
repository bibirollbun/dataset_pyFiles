!pip install catboost


import numpy as np
import pandas as pd
import os
import gc
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

import lightgbm as lgb

from catboost import CatBoostClassifier, Pool

from sklearn.metrics import roc_auc_score

from sklearn.base import BaseEstimator, TransformerMixin

from joblib import Parallel, delayed

from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings('ignore')



TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e12/test.csv"
ORG_PATH   = "/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv"
SAMPLE_SUB_PATH = "/kaggle/input/playground-series-s5e12/sample_submission.csv"


train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
org_df   = pd.read_csv(ORG_PATH)


common_cols = [col for col in train_df.columns if col in org_df.columns]

org_subset = org_df[common_cols].copy()

print(f"Original Train Shape: {train_df.shape}")
train_df = pd.concat([train_df, org_subset], axis=0, ignore_index=True)
print(f"New Train Shape (with Original Data): {train_df.shape}")


print(f"Train Shape: {train_df.shape}")
print(f"Test Shape : {test_df.shape}")
print(f"Original Data Shape: {org_df.shape}")


TARGET = "diagnosed_diabetes"

# We will separate them later using the 'id' column or index
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df[TARGET] = np.nan

print("\nSample of Original Data (The 'Magic' Source):")
display(org_df.head(3))


# Defining Numerical Columns for Clipping (ID and Target excluded)

num_cols = [
    c for c in train_df.columns 
    if train_df[c].dtype in ['float64', 'int64'] 
    and c not in ['id', TARGET, 'is_train']
]

for col in num_cols:
    lower = train_df[col].quantile(0.01)
    upper = train_df[col].quantile(0.99)
    
    train_df[col] = train_df[col].clip(lower, upper)
    test_df[col]  = test_df[col].clip(lower, upper)

print("Clipped outliers to 1st and 99th percentiles")


print("Generating 'Magic' Features from Original Data...")

BASE_COLS = [c for c in train_df.columns if c not in ['id', TARGET, 'is_train']]
ORIG_FEATURES = []

for col in BASE_COLS:
    mean_map = org_df.groupby(col)[TARGET].mean()
    new_mean_col = f"orig_mean_{col}"
    
    train_df[new_mean_col] = train_df[col].map(mean_map).fillna(org_df[TARGET].mean())
    test_df[new_mean_col]  = test_df[col].map(mean_map).fillna(org_df[TARGET].mean())
    ORIG_FEATURES.append(new_mean_col)

    count_map = org_df.groupby(col).size()
    new_count_col = f"orig_count_{col}"
    
    train_df[new_count_col] = train_df[col].map(count_map).fillna(0)
    test_df[new_count_col]  = test_df[col].map(count_map).fillna(0)
    ORIG_FEATURES.append(new_count_col)

print(f"--> Created {len(ORIG_FEATURES)} Magic Features.")



print("Generating Rounding Features...")
rounding_cols = ['triglycerides', 'cholesterol_total', 'systolic_bp']
rounding_levels = {
    '1s': 0,
    '10s': -1,
    '100s': -2
}

ROUND_FEATURES = []
for col in rounding_cols:
    for suffix, rnd_lvl in rounding_levels.items():
        new_col = f"{col}_rnd_{suffix}"
        ROUND_FEATURES.append(new_col)
        
        # Round and convert to int for both train and test
        train_df[new_col] = train_df[col].round(rnd_lvl).astype(int)
        test_df[new_col]  = test_df[col].round(rnd_lvl).astype(int)

print(f"--> Created {len(ROUND_FEATURES)} Rounding Features.")



print("Generating Frequency Features...")

freq_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
             'smoking_status', 'employment_status', 'diabetes_stage']

FREQ_FEATURES = []

for c in freq_cols:
    if c in train_df.columns:
        freq_map = train_df[c].value_counts(normalize=True)
        
        new_col = f"{c}_freq"
        
        train_df[new_col] = train_df[c].map(freq_map)
        test_df[new_col] = test_df[c].map(freq_map)
        
        train_df[new_col] = train_df[new_col].fillna(0)
        test_df[new_col] = test_df[new_col].fillna(0)
        
        FREQ_FEATURES.append(new_col)

print(f"--> Created {len(FREQ_FEATURES)} Frequency Features: {FREQ_FEATURES}")


print("Generating Medical Interaction & Binning Features...")

epsilon = 1e-6

# LDL / HDL Ratio (Key Heart Risk Marker)
train_df['ldl_hdl_ratio'] = train_df['ldl_cholesterol'] / (train_df['hdl_cholesterol'] + epsilon)
test_df['ldl_hdl_ratio']  = test_df['ldl_cholesterol'] / (test_df['hdl_cholesterol'] + epsilon)

# Triglycerides / HDL Ratio (Proxy for Insulin Resistance)
train_df['trig_hdl_ratio'] = train_df['triglycerides'] / (train_df['hdl_cholesterol'] + epsilon)
test_df['trig_hdl_ratio']  = test_df['triglycerides'] / (test_df['hdl_cholesterol'] + epsilon)

# Non-HDL Cholesterol
train_df['non_hdl'] = train_df['cholesterol_total'] - train_df['hdl_cholesterol']
test_df['non_hdl']  = test_df['cholesterol_total'] - test_df['hdl_cholesterol']

# BMI Group: Underweight (<18.5), Normal (18.5-25), Overweight (25-30), Obese (>30)
def bin_bmi(x):
    if x < 18.5: return 0
    elif x < 25.0: return 1
    elif x < 30.0: return 2
    else: return 3

if 'bmi' in train_df.columns:
    train_df['bmi_group'] = train_df['bmi'].apply(bin_bmi)
    test_df['bmi_group']  = test_df['bmi'].apply(bin_bmi)
else:
    print("Warning: BMI column not found, skipping bmi_group")

MEDICAL_FEATURES = [
    'ldl_hdl_ratio', 'trig_hdl_ratio', 'non_hdl', 'bmi_group'
]

MEDICAL_FEATURES = [c for c in MEDICAL_FEATURES if c in train_df.columns]

print(f"--> Created {len(MEDICAL_FEATURES)} Medical Features: {MEDICAL_FEATURES}")


cat_cols = [c for c in test_df.columns if test_df[c].dtype == 'object']

FEATURES = ORIG_FEATURES + num_cols + cat_cols + ROUND_FEATURES + FREQ_FEATURES + MEDICAL_FEATURES

print(f"\nTotal Features for Training: {len(FEATURES)}")


WIN_SIZE = 1000
THRESHOLD = 88
DRIFT_FEATURE = 'physical_activity_minutes_per_week'

print(f"Detecting drift in '{DRIFT_FEATURE}'...")

# Calculating Rolling Mean to smooth out noise and find the shift
rolling_mean = train_df[DRIFT_FEATURE].rolling(window=WIN_SIZE).mean()

# Finding the ID where the rolling mean crosses the threshold
cutoff_mask = rolling_mean > THRESHOLD
cutoff_id = rolling_mean[cutoff_mask].index.min()

print(f"--> Drift Cutoff Index found at: {cutoff_id}")



train_old = train_df.iloc[:cutoff_id].copy()
train_new = train_df.iloc[cutoff_id:].copy()

print(f"Old Data Size: {len(train_old)}")
print(f"New Data Size: {len(train_new)}")



VAL_WEIGHT_RATIO = 35
ORIG_WEIGHT_RATIO = 10

weights_old = np.ones(len(train_old))
weights_new = np.ones(len(train_new)) * VAL_WEIGHT_RATIO

sample_weights_full = np.concatenate([weights_old, weights_new])

num_original_rows = len(train_df) - len(sample_weights_full)
weights_original = np.ones(num_original_rows) * ORIG_WEIGHT_RATIO

sample_weights_full = np.concatenate([sample_weights_full, weights_original])

print(f"Sample Weights Created. Shape: {sample_weights_full.shape}")
print(f"Unique Weights: {np.unique(sample_weights_full)}")


# reseting index so weights align perfectly
X = pd.concat([train_old, train_new], axis=0).reset_index(drop=True)
y = X[TARGET]
X = X[FEATURES]

print("\nFinal Training Data Prepared:")
print(f"X Shape: {X.shape}")
print(f"y Shape: {y.shape}")


object_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"Found object columns to convert: {object_cols}")

if object_cols:
    for col in object_cols:
        X[col] = X[col].astype('category')
        test_df[col] = test_df[col].astype('category')
    print("âœ… Success: All object columns converted to 'category'.")
else:
    print("No object columns found (already converted).")


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_encode, aggs=['mean'], cv=5, smooth='auto', drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}

    def fit(self, X, y):
        temp_df = X.copy()
        temp_df['target'] = y
        for agg in self.aggs:
            self.global_stats_[agg] = y.agg(agg)
        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg in self.aggs:
                self.mappings_[col][agg] = temp_df.groupby(col)['target'].agg(agg)
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in self.cols_to_encode:
            for agg in self.aggs:
                new_col = f'TE_{col}_{agg}'
                map_series = self.mappings_[col][agg]
                X_out[new_col] = X[col].map(map_series).fillna(self.global_stats_[agg])
        if self.drop_original:
            X_out.drop(columns=self.cols_to_encode, inplace=True)
        return X_out

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)



def train_fold_parallel(fold, train_idx, val_idx, X, y, sample_weights, test_df, features, round_features):
    # GPU Allocation: Even folds -> GPU 0, Odd folds -> GPU 1
    gpu_id = fold % 2

    fold_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.015,
        "max_depth": 8,
        "subsample": 0.72,
        "colsample_bytree": 0.5,
        "alpha": 6.78,
        "reg_lambda": 1.13,
        "min_child_weight": 5,
        "tree_method": "hist",
        "device": f"cuda:{gpu_id}",
        "enable_categorical": True,
        "n_jobs": 4, 
        "verbosity": 0,
        "random_state": 42 + fold
    }

    # Data Slicing
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    w_train = sample_weights[train_idx]
    w_val   = sample_weights[val_idx]

    te = TargetEncoder(cols_to_encode=round_features, cv=5, smooth=1.0, drop_original=True)
    te.fit(X_train, y_train)
    
    X_train_encoded = te.transform(X_train)
    X_val_encoded   = te.transform(X_val)
    X_test_encoded  = te.transform(test_df[features])

    # Create DMatrices
    dtrain = xgb.DMatrix(X_train_encoded, label=y_train, weight=w_train, enable_categorical=True)
    dval   = xgb.DMatrix(X_val_encoded,   label=y_val,   weight=w_val,   enable_categorical=True)
    dtest  = xgb.DMatrix(X_test_encoded, enable_categorical=True)

    model = xgb.train(
        params=fold_params,
        dtrain=dtrain,
        num_boost_round=10000,
        evals=[(dtrain, 'train'), (dval, 'valid')],
        early_stopping_rounds=200,
        verbose_eval=False
    )

    best_iter = model.best_iteration
    valid_preds = model.predict(dval, iteration_range=(0, best_iter + 1))
    test_preds  = model.predict(dtest, iteration_range=(0, best_iter + 1))

    del dtrain, dval, dtest, model, X_train_encoded, X_val_encoded
    gc.collect()

    print(f"âœ… Fold {fold+1} Finished on GPU {gpu_id}. Best Iteration: {best_iter}")
    return val_idx, valid_preds, test_preds

print("Helper classes and worker function defined.")


FOLDS = 10
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

print(f"ğŸš€ Launching {FOLDS} folds across 2 GPUs in parallel...")

results = Parallel(n_jobs=2, backend="threading")(
    delayed(train_fold_parallel)(
        fold, train_idx, val_idx,
        X, y, sample_weights_full, test_df, FEATURES, ROUND_FEATURES
    )
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y))
)

print("\n Parallel Training Complete!")


oof_preds_full = np.zeros(len(X))
test_preds_final = np.zeros(len(test_df))

print("Aggregating results from parallel workers")


for val_idx, valid_preds, fold_test_preds in results:
    # Map OOF predictions back to their original indices
    oof_preds_full[val_idx] = valid_preds
    test_preds_final += fold_test_preds / FOLDS


local_auc = roc_auc_score(y, oof_preds_full)
print(f"\n>>> FINAL OVERALL OOF AUC: {local_auc:.5f} <<<")

np.save("oof_preds_35x.npy", oof_preds_full)
np.save("test_preds_35x.npy", test_preds_final)
print("Saved raw predictions to .npy files.")



def train_lgbm_parallel(fold, train_idx, val_idx, X, y, sample_weights, test_df, features, round_features):
    # Even folds -> GPU 0, Odd folds -> GPU 1
    gpu_id = fold % 2
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.01,
        'n_estimators': 8000,
        'max_depth': 8,
        'num_leaves': 64,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'reg_alpha': 0.1,
        'reg_lambda': 5.0,
        'device': 'gpu',
        'gpu_device_id': gpu_id,
        'n_jobs': 4,
        'verbose': -1,
        'seed': 42 + fold
    }

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    w_train = sample_weights[train_idx]
    w_val   = sample_weights[val_idx]

    te = TargetEncoder(cols_to_encode=round_features, cv=5, smooth=1.0, drop_original=True)
    te.fit(X_train, y_train)
    
    X_train_encoded = te.transform(X_train)
    X_val_encoded   = te.transform(X_val)
    X_test_encoded  = te.transform(test_df[features])

    # Create LightGBM Datasets
    # Pass 'weight' directly to the Dataset constructor
    ds_train = lgb.Dataset(X_train_encoded, label=y_train, weight=w_train)
    ds_val   = lgb.Dataset(X_val_encoded,   label=y_val,   weight=w_val, reference=ds_train)

    callbacks = [
        lgb.early_stopping(stopping_rounds=200, verbose=False),
        lgb.log_evaluation(period=0)
    ]
    
    model = lgb.train(
        lgb_params,
        ds_train,
        valid_sets=[ds_train, ds_val],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )

    best_iter = model.best_iteration
    valid_preds = model.predict(X_val_encoded, num_iteration=best_iter)
    test_preds  = model.predict(X_test_encoded, num_iteration=best_iter)

    print(f"âœ… LGBM Fold {fold+1} Finished on GPU {gpu_id}. Best Iter: {best_iter}")
    return val_idx, valid_preds, test_preds


print(f" Launching LightGBM Ensemble ({FOLDS} folds) across 2 GPUs...")

lgbm_results = Parallel(n_jobs=2, backend="threading")(
    delayed(train_lgbm_parallel)(
        fold, train_idx, val_idx,
        X, y, sample_weights_full, test_df, FEATURES, ROUND_FEATURES
    )
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y))
)

print("\n LightGBM Training Complete!")


def train_cat_parallel(fold, train_idx, val_idx, X, y, sample_weights, test_df, features, round_features):
    gpu_id = fold % 2
    
    cat_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'learning_rate': 0.01,
        'iterations': 8000,
        'depth': 6,
        'l2_leaf_reg': 3,
        'task_type': 'GPU',
        'devices': str(gpu_id),
        'verbose': 0,
        'random_seed': 42 + fold,
        'allow_writing_files': False
    }

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    w_train = sample_weights[train_idx]
    w_val   = sample_weights[val_idx]

    te = TargetEncoder(cols_to_encode=round_features, cv=5, smooth=1.0, drop_original=True)
    te.fit(X_train, y_train)
    
    X_train_encoded = te.transform(X_train)
    X_val_encoded   = te.transform(X_val)
    X_test_encoded  = te.transform(test_df[features])

    cat_features_indices = [i for i, col in enumerate(X_train_encoded.columns) 
                            if X_train_encoded[col].dtype.name == 'category']

    train_pool = Pool(X_train_encoded, y_train, cat_features=cat_features_indices, weight=w_train)
    val_pool   = Pool(X_val_encoded, y_val, cat_features=cat_features_indices, weight=w_val)
    test_pool  = Pool(X_test_encoded, cat_features=cat_features_indices)

    model = CatBoostClassifier(**cat_params)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=200, verbose=False)

    best_iter = model.get_best_iteration()
    valid_preds = model.predict_proba(val_pool)[:, 1]
    test_preds  = model.predict_proba(test_pool)[:, 1]

    print(f"ğŸ�± CatBoost Fold {fold+1} Finished on GPU {gpu_id}. Best Iter: {best_iter}")
    return val_idx, valid_preds, test_preds


print(f"ğŸš€ Launching CatBoost Ensemble ({FOLDS} folds)...")

cat_results = Parallel(n_jobs=2, backend="loky")(
    delayed(train_cat_parallel)(
        fold, train_idx, val_idx,
        X, y, sample_weights_full, test_df, FEATURES, ROUND_FEATURES
    )
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y))
)

print("\n CatBoost Training Complete!")


lgbm_oof = np.zeros(len(X))
lgbm_test = np.zeros(len(test_df))
cat_oof = np.zeros(len(X))
cat_test = np.zeros(len(test_df))

for val_idx, valid_preds, fold_test_preds in lgbm_results:
    lgbm_oof[val_idx] = valid_preds
    lgbm_test += fold_test_preds / FOLDS

for val_idx, valid_preds, fold_test_preds in cat_results:
    cat_oof[val_idx] = valid_preds
    cat_test += fold_test_preds / FOLDS

xgb_oof = oof_preds_full
xgb_test = test_preds_final

print(f"XGBoost OOF  : {roc_auc_score(y, xgb_oof):.5f}")
print(f"LightGBM OOF : {roc_auc_score(y, lgbm_oof):.5f}")
print(f"CatBoost OOF : {roc_auc_score(y, cat_oof):.5f}")


def get_auc_3way(weights):
    w = np.exp(weights) / np.sum(np.exp(weights))
    blend = w[0]*xgb_oof + w[1]*lgbm_oof + w[2]*cat_oof
    return -roc_auc_score(y, blend)

init_weights = [1/3, 1/3, 1/3]
res = minimize(get_auc_3way, init_weights, method='Nelder-Mead')

final_w = np.exp(res.x) / np.sum(np.exp(res.x))

print(f"\nâœ… Optimal Weights:")
print(f"   XGB: {final_w[0]:.4f} | LGB: {final_w[1]:.4f} | CAT: {final_w[2]:.4f}")



# Weighted Blend
collab_oof = final_w[0]*xgb_oof + final_w[1]*lgbm_oof + final_w[2]*cat_oof
collab_test = final_w[0]*xgb_test + final_w[1]*lgbm_test + final_w[2]*cat_test

print(f">>> FINAL 3-WAY ENSEMBLE SCORE: {roc_auc_score(y, collab_oof):.5f} <<<")



print("\nCalibration in progress...")

iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(collab_oof, y)

final_calibrated_preds = iso_reg.predict(collab_test)

print(f"Original Mean Pred: {collab_test.mean():.4f}")
print(f"Calibrated Mean Pred: {final_calibrated_preds.mean():.4f}")


submission = pd.read_csv(SAMPLE_SUB_PATH)
submission[TARGET] = final_calibrated_preds
submission.to_csv("submission.csv", index=False)

print("\nâœ… Final Calibrated Submission Saved as 'submission.csv'!")

