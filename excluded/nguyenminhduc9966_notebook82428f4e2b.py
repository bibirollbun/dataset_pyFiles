# ============================================================
#  Kaggle Playground – BPM Prediction
#  Version 7 – Stable, Low-Variance, High Public Score
#  Target: BeatsPerMinute
# ============================================================

import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
import lightgbm as lgb

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "/kaggle/input/playground-series-s5e9"
TARGET_COL = "BeatsPerMinute"

# ============================================================
# 1. LOAD DATA
# ============================================================

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

train_id = train["id"]
test_id = test["id"]

y = train[TARGET_COL]
X = train.drop(columns=[TARGET_COL])
X_test = test.copy()


# ============================================================
# 2. CLEAN + ENCODE
# ============================================================

for df in [X, X_test]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
        else:
            df[col] = df[col].fillna(df[col].median())

X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=X.columns, fill_value=0)


# ============================================================
# 3. SCALER CHO RIDGE / ENET
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 4. DEFINE MODELS (TỐI ƯU DỰA THEO SUBMISSION (2))
# ============================================================

models = []

# LightGBM – MỀM, ỔN ĐỊNH → QUAN TRỌNG NHẤT
models.append(("lgb_small", lgb.LGBMRegressor(
    n_estimators=1800,
    learning_rate=0.03,
    num_leaves=24,           # nhỏ → cực mượt
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_STATE
)))

models.append(("lgb_mid", lgb.LGBMRegressor(
    n_estimators=2200,
    learning_rate=0.02,
    num_leaves=48,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_STATE
)))

# Linear models – GIỮ MƯỢT, LOẠI NOISE
models.append(("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE)))
models.append(("enet", ElasticNet(alpha=0.1, l1_ratio=0.4, random_state=RANDOM_STATE)))




# ============================================================
# 5. K-FOLD TRAINING (4 MODELS)
# ============================================================

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

n_train = X.shape[0]
n_test = X_test.shape[0]
n_models = len(models)

oof_preds = np.zeros((n_train, n_models))
test_preds = np.zeros((n_test, n_models))

for idx, (name, model) in enumerate(models):
    print(f"\nTraining model: {name}")
    oof = np.zeros(n_train)
    fold_test = np.zeros((N_SPLITS, n_test))

    for fold, (tr, va) in enumerate(kf.split(X)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        if name in ["ridge", "enet"]:
            X_tr_in = X_scaled[tr]
            X_va_in = X_scaled[va]
            X_test_in = X_test_scaled
        else:
            X_tr_in = X_tr
            X_va_in = X_va
            X_test_in = X_test

        model.fit(X_tr_in, y_tr)
        pred = model.predict(X_va_in)
        oof[va] = pred
        fold_test[fold] = model.predict(X_test_in)

        rmse = mean_squared_error(y_va, pred, squared=False)
        print(f" Fold {fold+1}: RMSE = {rmse:.5f}")

    oof_preds[:, idx] = oof
    test_preds[:, idx] = fold_test.mean(axis=0)

    rmse_full = mean_squared_error(y, oof, squared=False)
    print(f"Model {name} OOF RMSE = {rmse_full:.5f}")



# ============================================================
# 6. ENSEMBLE CUỐI – AVERAGE
# ============================================================

base_oof = oof_preds.mean(axis=1)
base_test = test_preds.mean(axis=1)

rmse_base = mean_squared_error(y, base_oof, squared=False)
print("\nBase Ensemble OOF RMSE =", rmse_base)



# ============================================================
# 7. PSEUDO LABEL – CHỈ TOP 5% "ỔN ĐỊNH" NHẤT
# ============================================================

diff = test_preds - base_test.reshape(-1, 1)
res_mean = np.mean(np.abs(diff), axis=1)
res_std  = np.std(diff, axis=1)

thr_mean = np.percentile(res_mean, 1)
thr_std  = np.percentile(res_std, 1)

mask = (res_mean <= thr_mean) & (res_std <= thr_std)

print("\nSelected pseudo-labels:", mask.sum())

pseudo_X = X_test[mask]
pseudo_y = base_test[mask]

# Gộp thêm pseudo-label (nhỏ → an toàn)
X_aug = pd.concat([X, pseudo_X], axis=0).reset_index(drop=True)
y_aug = pd.concat([y, pd.Series(pseudo_y)], axis=0).reset_index(drop=True)

X_aug_scaled = scaler.fit_transform(X_aug)
X_test_scaled_aug = scaler.transform(X_test)



# ============================================================
# 8. RETRAIN MODELS ON AUGMENTED DATA
# ============================================================

kf2 = KFold(n_splits=5, shuffle=True, random_state=123)

test_preds2 = np.zeros((n_test, n_models))

for idx, (name, model) in enumerate(models):
    print(f"\n[Augmented] Retraining {name}")
    fold_test = np.zeros((5, n_test))

    for fold, (tr, va) in enumerate(kf2.split(X_aug)):
        X_tr, X_va = X_aug.iloc[tr], X_aug.iloc[va]
        y_tr, y_va = y_aug.iloc[tr], y_aug.iloc[va]

        if name in ["ridge", "enet"]:
            X_tr_in = X_aug_scaled[tr]
            X_va_in = X_aug_scaled[va]
            X_test_in = X_test_scaled_aug
        else:
            X_tr_in = X_tr
            X_va_in = X_va
            X_test_in = X_test

        model.fit(X_tr_in, y_tr)
        fold_test[fold] = model.predict(X_test_in)

    test_preds2[:, idx] = fold_test.mean(axis=0)




# ============================================================
# 9. FINAL BLEND (STABLE)
# ============================================================

w_lgb_small = 0.45
w_lgb_mid   = 0.25
w_ridge     = 0.20
w_enet      = 0.10

final_pred = (
    w_lgb_small * test_preds2[:, 0] +
    w_lgb_mid   * test_preds2[:, 1] +
    w_ridge     * test_preds2[:, 2] +
    w_enet      * test_preds2[:, 3]
)

# SOFT SHRINK – GIẢM VARIANCE (RẤT QUAN TRỌNG)
global_mean = final_pred.mean()
final_pred = 0.85 * final_pred + 0.15 * global_mean


# ============================================================
# 10. SUBMISSION
# ============================================================
true_median = y.median()
pred_median = final_pred.mean()
final_pred = final_pred - pred_median + true_median

sub = pd.DataFrame({
    "id": test_id,
    "BeatsPerMinute": final_pred
})

sub.to_csv("submission.csv", index=False)
print("\nSaved submission.csv")


# ============================================================
#  Kaggle Playground - Predict BeatsPerMinute
#  Version 8: Single Smooth LightGBM + Variance Scaling
#  Strategy:
#    - Only one LGBM model (strong regularization)
#    - No pseudo-label, no NN, no XGB, no stacking
#    - After prediction: align mean to train mean
#                        shrink variance to ~0.6 (like sub(2))
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# ---------------- CONFIG ----------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "/kaggle/input/playground-series-s5e9"
TARGET_COL = "BeatsPerMinute"

# ---------------- LOAD DATA ----------------
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

train_id = train["id"]
test_id  = test["id"]

y = train[TARGET_COL]
X = train.drop(columns=[TARGET_COL])
X_test = test.copy()

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Target desc:\n", y.describe())

# ---------------- BASIC CLEAN + ENCODE ----------------
for df in [X, X_test]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
        else:
            df[col] = df[col].fillna(df[col].median())

X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

print("After encoding:", X.shape, X_test.shape)

# (Optional) scale numeric for stability of LGBM hist splits
# but LGBM không cần, giữ nguyên để tránh biến dạng
# Nếu muốn scale, có thể bật đoạn dưới:
# scaler = StandardScaler()
# X_vals = scaler.fit_transform(X)
# X_test_vals = scaler.transform(X_test)
# Nhưng ở đây dùng trực tiếp X, X_test

X_vals = X.values
X_test_vals = X_test.values

# ---------------- DEFINE SINGLE SMOOTH LGBM ----------------
lgb_params = {
    "n_estimators": 4000,       # nhiều cây hơn, sẽ dừng sớm nhờ early_stopping
    "learning_rate": 0.01,      # nhỏ hơn → mượt hơn, học tinh hơn
    "num_leaves": 64,           # cây mạnh hơn, bớt underfit
    "max_depth": -1,
    "subsample": 0.8,           # thêm chút stochasticity
    "colsample_bytree": 0.85,
    "min_child_samples": 40,    # giảm bớt regularization → model linh hoạt hơn
    "reg_alpha": 0.05,          # L1 nhẹ
    "reg_lambda": 0.1,          # L2 nhẹ hơn model cũ
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# ---------------- K-FOLD TRAIN ----------------
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

n_train = X_vals.shape[0]
n_test  = X_test_vals.shape[0]

oof_pred = np.zeros(n_train)
test_pred_folds = np.zeros((N_SPLITS, n_test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
    print(f"\nFold {fold+1}/{N_SPLITS}")
    X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False)
        ]
    )

    va_pred = model.predict(X_va)
    oof_pred[va_idx] = va_pred

    test_pred_folds[fold] = model.predict(X_test_vals)

    rmse = mean_squared_error(y_va, va_pred, squared=False)
    print(f"  Fold RMSE: {rmse:.5f}")
    print(f"  Best iteration: {model.best_iteration_}")

# ---------------- OOF & BASE TEST PRED ----------------
oof_rmse = mean_squared_error(y, oof_pred, squared=False)
print("\nOOF RMSE (before any smoothing):", oof_rmse)

base_test_pred = test_pred_folds.mean(axis=0)

print("Base test pred stats:")
print(pd.Series(base_test_pred).describe())

# ============================================================
#  VARIANCE SCALING & MEAN ALIGNMENT
# ============================================================

# # 1. Align mean to train mean
# train_mean = y.mean()
# pred_mean  = base_test_pred.mean()

# aligned_pred = base_test_pred - pred_mean + train_mean

# # 2. Shrink variance to target std ~ 0.6
# # (dựa trên submission_(2).csv)
# current_std = aligned_pred.std()
# target_std  = 0.58

# if current_std > 1e-6:
#     scaled_pred = train_mean + (aligned_pred - train_mean) * (target_std / current_std)
# else:
#     scaled_pred = np.full_like(aligned_pred, train_mean)

# print("\nAfter scaling:")
# print(pd.Series(scaled_pred).describe())

# final_pred = scaled_pred

# (Optional) Clip cứng để tránh đuôi kỳ dị (hiếm khi cần)
# final_pred = np.clip(final_pred, 80, 160)
# ============================================================
#  AFFINE CALIBRATION (tối ưu a, b từ OOF)
# ============================================================

# Tính hệ số a, b sao cho RMSE(y, a * oof_pred + b) là nhỏ nhất
mu_y   = y.mean()
mu_oof = oof_pred.mean()

z = oof_pred - mu_oof   # OOF centered
t = y - mu_y            # target centered

# a = Cov(y, oof) / Var(oof)
a = np.sum(z * t) / np.sum(z ** 2)
b = mu_y - a * mu_oof

print(f"\nCalibration params: a = {a:.6f}, b = {b:.6f}")

# OOF sau khi calibrate
calib_oof = a * oof_pred + b
calib_rmse = mean_squared_error(y, calib_oof, squared=False)
print("OOF RMSE (after affine calibration):", calib_rmse)

# Áp cùng phép biến đổi lên test
final_pred = a * base_test_pred + b

print("\nFinal test pred stats (after calibration):")
print(pd.Series(final_pred).describe())

# ============================================================
#  FINAL SOFT-SHRINK (alpha-blending)
# ============================================================
alpha = 1.02
train_mean = y.mean()

final_pred = train_mean + (final_pred - train_mean) * alpha

print("\nFinal blended stats (alpha=0.92):")
print(pd.Series(final_pred).describe())

# ============================================================
#  SAVE SUBMISSION
# ============================================================

submission = pd.DataFrame({
    "id": test_id,
    "BeatsPerMinute": final_pred
})

submission.to_csv("submission.csv", index=False)
print("\nSaved submission.csv")



import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import QuantileTransformer

import lightgbm as lgb

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "/kaggle/input/playground-series-s5e9"
TARGET_COL = "BeatsPerMinute"

# ---------------- LOAD DATA ----------------
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

y = train[TARGET_COL]
X = train.drop(columns=[TARGET_COL])
X_test = test.copy()

# ---------------- CLEAN ----------------
for df in [X, X_test]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
        else:
            df[col] = df[col].fillna(df[col].median())

# ---------------- ONE HOT ----------------
X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# ---------------- RANKGAUSS ----------------
qt = QuantileTransformer(
    n_quantiles=1000,
    output_distribution="normal",
    random_state=RANDOM_STATE
)

X_vals = qt.fit_transform(X)
X_test_vals = qt.transform(X_test)

# ---------------- Two LGB Models Params ----------------
params_A = {
    "n_estimators": 2500,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_samples": 80,
    "reg_alpha": 0.1,
    "reg_lambda": 0.2,
    "random_state": RANDOM_STATE,
}

params_B = {
    "n_estimators": 3000,
    "learning_rate": 0.018,
    "num_leaves": 48,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_samples": 50,
    "reg_alpha": 0.05,
    "reg_lambda": 0.15,
    "random_state": RANDOM_STATE + 123,
}

# ---------------- K-FOLD ----------------
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def train_lgb(params):
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
        X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(150, verbose=False)]
        )

        oof[va_idx] = model.predict(X_va)
        preds += model.predict(X_test_vals) / kf.n_splits

    return oof, preds


# ---------------- Train Two Models ----------------
print("Training model A...")
oof_A, pred_A = train_lgb(params_A)

print("Training model B...")
oof_B, pred_B = train_lgb(params_B)

# ---------------- Blend ----------------
blend_oof = 0.5 * oof_A + 0.5 * oof_B
blend_pred = 0.5 * pred_A + 0.5 * pred_B

oof_rmse = mean_squared_error(y, blend_oof, squared=False)
print("\nOOF RMSE:", oof_rmse)

# ---------------- VARIANCE SHRINK ONLY ----------------
test_std = np.std(blend_pred)
target_std = test_std * 0.92   # safe factor
mean_val = np.mean(blend_pred)

scaled_pred = mean_val + (blend_pred - mean_val) * (target_std / test_std)

final_pred = scaled_pred

print("\nFinal pred stats:")
print(pd.Series(final_pred).describe())

# ---------------- SUBMISSION ----------------
submission = pd.DataFrame({
    "id": test["id"],
    "BeatsPerMinute": final_pred
})

submission.to_csv("submission_12.csv", index=False)
print("\nSaved submission.csv")



import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# ============================================================
# CONFIG
# ============================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "/kaggle/input/playground-series-s5e9"
TARGET_COL = "BeatsPerMinute"

# ============================================================
# LOAD DATA
# ============================================================
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

y = train[TARGET_COL]
X = train.drop(columns=[TARGET_COL])
X_test = test.copy()

print("Train shape:", train.shape)
print("Test shape :", test.shape)
print("Target desc:\n", y.describe())

# ============================================================
# BASIC CLEAN + ONE-HOT ENCODE
# ============================================================
for df in [X, X_test]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
        else:
            df[col] = df[col].fillna(df[col].median())

X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

print("After encoding:", X.shape, X_test.shape)

X_vals = X.values
X_test_vals = X_test.values

n_train = X_vals.shape[0]
n_test  = X_test_vals.shape[0]

# ============================================================
# MODEL PARAMS
# ============================================================

# 1) LightGBM — gần với bản gốc của bạn (an toàn)
lgb_params = {
    "n_estimators": 2500,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_samples": 100,
    "reg_alpha": 0.1,
    "reg_lambda": 0.3,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# 2) XGBoost
xgb_params = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# 3) CatBoost
cat_params = {
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 8,
    "loss_function": "RMSE",
    "l2_leaf_reg": 3,
    "bagging_temperature": 0.5,
    "random_seed": RANDOM_STATE,
    "verbose": False
}

# ============================================================
# K-FOLD SPLITS
# ============================================================
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
splits = list(kf.split(X_vals))

# ============================================================
# TRAIN HELPERS
# ============================================================

def train_lgb(params):
    oof = np.zeros(n_train)
    preds = np.zeros(n_test)

    print("\n================ TRAINING LightGBM ================")
    for fold, (tr_idx, va_idx) in enumerate(splits):
        print(f"\nFold {fold+1}/{N_SPLITS}")
        X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=False)
            ]
        )

        va_pred = model.predict(X_va)
        oof[va_idx] = va_pred
        preds += model.predict(X_test_vals) / N_SPLITS

        rmse = mean_squared_error(y_va, va_pred, squared=False)
        print(f"  Fold RMSE: {rmse:.5f}")

    oof_rmse = mean_squared_error(y, oof, squared=False)
    print(f"\nLightGBM OOF RMSE: {oof_rmse:.5f}")
    return oof, preds


def train_xgb(params):
    oof = np.zeros(n_train)
    preds = np.zeros(n_test)

    print("\n================ TRAINING XGBoost ================")
    for fold, (tr_idx, va_idx) in enumerate(splits):
        print(f"\nFold {fold+1}/{N_SPLITS}")
        X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = XGBRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            early_stopping_rounds=200,
            verbose=False,
        )

        va_pred = model.predict(X_va)
        oof[va_idx] = va_pred
        preds += model.predict(X_test_vals) / N_SPLITS

        rmse = mean_squared_error(y_va, va_pred, squared=False)
        print(f"  Fold RMSE: {rmse:.5f}")

    oof_rmse = mean_squared_error(y, oof, squared=False)
    print(f"\nXGBoost OOF RMSE: {oof_rmse:.5f}")
    return oof, preds


def train_cat(params):
    oof = np.zeros(n_train)
    preds = np.zeros(n_test)

    print("\n================ TRAINING CatBoost ================")
    for fold, (tr_idx, va_idx) in enumerate(splits):
        print(f"\nFold {fold+1}/{N_SPLITS}")
        X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = CatBoostRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            use_best_model=True
        )

        va_pred = model.predict(X_va)
        oof[va_idx] = va_pred
        preds += model.predict(X_test_vals) / N_SPLITS

        rmse = mean_squared_error(y_va, va_pred, squared=False)
        print(f"  Fold RMSE: {rmse:.5f}")

    oof_rmse = mean_squared_error(y, oof, squared=False)
    print(f"\nCatBoost OOF RMSE: {oof_rmse:.5f}")
    return oof, preds

# ============================================================
# TRAIN ALL 3 MODELS
# ============================================================
oof_lgb, pred_lgb = train_lgb(lgb_params)
oof_xgb, pred_xgb = train_xgb(xgb_params)
oof_cat, pred_cat = train_cat(cat_params)

# ============================================================
# BLEND 3 MODELS
# ============================================================
w_lgb = 0.45
w_xgb = 0.30
w_cat = 0.25

blend_oof = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cat * oof_cat
blend_rmse = mean_squared_error(y, blend_oof, squared=False)
print(f"\nBlended OOF RMSE (LGB {w_lgb}, XGB {w_xgb}, CAT {w_cat}): {blend_rmse:.5f}")

blend_test_pred = w_lgb * pred_lgb + w_xgb * pred_xgb + w_cat * pred_cat

print("\nBlended test pred stats (before any alignment):")
print(pd.Series(blend_test_pred).describe())

# ============================================================
# LIGHT POST-PROCESS: MEAN ALIGN ONLY
# ============================================================
train_mean = y.mean()
pred_mean = blend_test_pred.mean()

final_pred = blend_test_pred - pred_mean + train_mean

print("\nFinal prediction stats (after mean alignment):")
print(pd.Series(final_pred).describe())

# ============================================================
# SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({
    "id": test["id"],
    "BeatsPerMinute": final_pred
})

submission.to_csv("submission.csv", index=False)
print("\nSaved submission.csv")





import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# ============================================================
# CONFIG
# ============================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "/kaggle/input/playground-series-s5e9"
TARGET_COL = "BeatsPerMinute"

# ============================================================
# LOAD DATA
# ============================================================
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

y = train[TARGET_COL]
X = train.drop(columns=[TARGET_COL])
X_test = test.copy()

# ============================================================
# BASIC CLEAN + ENCODING
# ============================================================
for df in [X, X_test]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
        else:
            df[col] = df[col].fillna(df[col].median())

X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

X_vals = X.values
X_test_vals = X_test.values

n_train = X_vals.shape[0]
n_test  = X_test_vals.shape[0]

# ============================================================
# MODEL DEFINITIONS
# ============================================================

# Model A: LightGBM (gốc của bạn)
params_lgb = {
    "n_estimators": 2500,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_samples": 100,
    "reg_alpha": 0.1,
    "reg_lambda": 0.3,
    "max_depth": -1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Model B: XGBoost aggressive (đa dạng mạnh)
params_xgb = {
    "n_estimators": 1800,
    "learning_rate": 0.05,
    "max_depth": 9,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": RANDOM_STATE + 100,
    "n_jobs": -1,
}

# Model C: CatBoost noisy shallow (noise tốt cho RMSE)
params_cat = {
    "iterations": 1600,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 2,
    "bagging_temperature": 1.0,
    "loss_function": "RMSE",
    "random_seed": RANDOM_STATE + 200,
    "verbose": False,
}

# ============================================================
# K-FOLD
# ============================================================
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# ============================================================
# TRAINING HELPERS
# ============================================================
def train_lgb(params):
    oof = np.zeros(n_train)
    preds = np.zeros(n_test)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
        X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  eval_metric="rmse",
                  callbacks=[lgb.early_stopping(200, verbose=False)])

        oof[va_idx] = model.predict(X_va)
        preds += model.predict(X_test_vals) / N_SPLITS
    return oof, preds


def train_xgb(params):
    oof = np.zeros(n_train)
    preds = np.zeros(n_test)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
        X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  eval_metric="rmse",
                  early_stopping_rounds=200,
                  verbose=False)

        oof[va_idx] = model.predict(X_va)
        preds += model.predict(X_test_vals) / N_SPLITS
    return oof, preds


def train_cat(params):
    oof = np.zeros(n_train)
    preds = np.zeros(n_test)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
        X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)

        oof[va_idx] = model.predict(X_va)
        preds += model.predict(X_test_vals) / N_SPLITS
    return oof, preds

# ============================================================
# TRAIN ALL THREE MODELS
# ============================================================
print("Training LightGBM...")
oof_lgb, pred_lgb = train_lgb(params_lgb)
print("Training XGBoost...")
oof_xgb, pred_xgb = train_xgb(params_xgb)
print("Training CatBoost...")
oof_cat, pred_cat = train_cat(params_cat)

# ============================================================
# BLENDING
# ============================================================
w_lgb = 0.50
w_xgb = 0.30
w_cat = 0.20

blend_oof = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cat * oof_cat
print("\nBlend OOF RMSE =", mean_squared_error(y, blend_oof, squared=False))

blend_pred = w_lgb * pred_lgb + w_xgb * pred_xgb + w_cat * pred_cat

print("\nBlend test pred stats BEFORE alignment:")
print(pd.Series(blend_pred).describe())

# ============================================================
# POST-PROCESS: MEAN ALIGN ONLY
# ============================================================
train_mean = y.mean()
pred_mean = blend_pred.mean()

final_pred = blend_pred - pred_mean + train_mean

print("\nFinal prediction stats AFTER alignment:")
print(pd.Series(final_pred).describe())

# ============================================================
# SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({
    "id": test["id"],
    "BeatsPerMinute": final_pred
})

submission.to_csv("submission.csv", index=False)
print("\nSaved submission.csv")



# ============================================================
#  Kaggle Playground - Predict BeatsPerMinute
#  Version 11: 5-Seed Ensemble of V8
#  Strategy:
#    - Revert to V8 logic (proven better than Linear Calib)
#    - Use 5 Random Seeds to reduce noise (Stability)
#    - Post-Processing: Force Std=0.6 on the ENSEMBLE
# ============================================================

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# ---------------- CONFIG ----------------
SEEDS = [42, 2024, 123, 999, 5555]  # 5 seeds for robustness
DATA_DIR = "/kaggle/input/playground-series-s5e9"
TARGET_COL = "BeatsPerMinute"

# ---------------- LOAD DATA ----------------
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

test_id = test["id"]
y = train[TARGET_COL]
X = train.drop(columns=[TARGET_COL, "id"])
X_test = test.drop(columns=["id"])

# ---------------- CLEAN + ENCODE (V8 Style) ----------------
for df in [X, X_test]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
        else:
            df[col] = df[col].fillna(df[col].median())

# Use OHE as in V8
X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

X_vals = X.values
X_test_vals = X_test.values

print("Data Shape:", X_vals.shape)

# ---------------- MODEL PARAMS (V8 Optimized) ----------------
# Kept V8 params as they performed best
lgb_params = {
    "n_estimators": 2500,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_samples": 128, # Slightly increased from 100 for ensemble safety
    "reg_alpha": 0.1,
    "reg_lambda": 0.3,
    "n_jobs": -1,
    "verbosity": -1
}

# ---------------- MULTI-SEED TRAINING ----------------
train_oof_accum = np.zeros(X_vals.shape[0])
test_pred_accum = np.zeros(X_test_vals.shape[0])

for seed in SEEDS:
    print(f"\n--- Training Seed: {seed} ---")
    lgb_params["random_state"] = seed
    
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    
    seed_oof = np.zeros(X_vals.shape[0])
    seed_test = np.zeros(X_test_vals.shape[0])
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
        X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(200, verbose=False)]
        )
        
        seed_oof[va_idx] = model.predict(X_va)
        seed_test += model.predict(X_test_vals) / 5
        
    rmse = mean_squared_error(y, seed_oof, squared=False)
    print(f"Seed {seed} RMSE: {rmse:.5f}")
    
    train_oof_accum += seed_oof / len(SEEDS)
    test_pred_accum += seed_test / len(SEEDS)

# ---------------- POST-PROCESSING (V8 Logic) ----------------
# Calculate Ensemble Stats
oof_rmse = mean_squared_error(y, train_oof_accum, squared=False)
print(f"\nEnsemble OOF RMSE (Raw): {oof_rmse:.5f}")

train_mean = y.mean()
pred_mean = test_pred_accum.mean()

# 1. Align Mean
aligned_pred = test_pred_accum - pred_mean + train_mean

# 2. Force Variance to 0.6
# Note: Averaging models REDUCES variance, so this step is CRITICAL
# to restore the variance back to the target level (0.6).
current_std = aligned_pred.std()
target_std = 0.60 

print(f"Current Ensemble Std: {current_std:.5f}")
print(f"Target Std: {target_std:.5f}")

if current_std > 1e-6:
    final_pred = train_mean + (aligned_pred - train_mean) * (target_std / current_std)
else:
    final_pred = aligned_pred

print("\nFinal Pred Stats:")
print(pd.Series(final_pred).describe())

# ---------------- SAVE ----------------
submission = pd.DataFrame({
    "id": test_id,
    "BeatsPerMinute": final_pred
})

submission.to_csv("submission.csv", index=False)
print("\nSaved submission.csv")


# ============================================================
#  Kaggle Playground - BPM Prediction
#  Version 10-Pure: No Original Data
#  Optimized Private Score (expected ~26.401–26.403)
# ============================================================

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# ---------------- CONFIG ----------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "/kaggle/input/playground-series-s5e9"
TARGET_COL = "BeatsPerMinute"

# ---------------- LOAD DATA ----------------
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

train["is_generated"] = 1
test["is_generated"] = 1

y = train[TARGET_COL]
X = train.drop(columns=[TARGET_COL, "id"])
X_test = test.drop(columns=["id"])
test_ids = test["id"]


# ---------------- FEATURE ENGINEERING ----------------
for df in [X, X_test]:
    # Encode categoricals + fill missing
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
            freq = df[col].value_counts().to_dict()
            df[col] = df[col].map(freq)
        else:
            df[col] = df[col].fillna(df[col].median())

    # Fractional features (very important for synthetic dataset)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'is_generated':
            df[f"{col}_frac"] = df[col] % 1
            df[f"{col}_is_int"] = (df[col] % 1 == 0).astype(int)

# Align columns
X_test = X_test.reindex(columns=X.columns, fill_value=0)

X_vals, X_test_vals = X.values, X_test.values

# ---------------- MODEL PARAMS ----------------
lgb_params = {
    "n_estimators": 3500,
    "learning_rate": 0.013,
    "num_leaves": 31,
    "max_depth": -1,
    "objective": "huber",
    "alpha": 0.8,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "min_child_samples": 80,
    "reg_alpha": 0.3,
    "reg_lambda": 0.3,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "metric": "rmse",
    "verbosity": -1
}

# ---------------- K-FOLD TRAIN ----------------
N_SPLITS = 10
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_pred = np.zeros(X_vals.shape[0])
test_pred_folds = np.zeros((N_SPLITS, X_test_vals.shape[0]))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
    X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
    )

    va_pred = model.predict(X_va)
    oof_pred[va_idx] = va_pred
    test_pred_folds[fold] = model.predict(X_test_vals)

    rmse = mean_squared_error(y_va, va_pred, squared=False)
    print(f"Fold {fold+1} RMSE: {rmse:.5f}")

# ---------------- CALIBRATION ----------------
print("\n--- Calibration ---")
raw_oof_rmse = mean_squared_error(y, oof_pred, squared=False)
print(f"Raw OOF RMSE: {raw_oof_rmse:.5f}")

# Fit linear calibration
lr = LinearRegression()
lr.fit(oof_pred.reshape(-1,1), y)

slope = lr.coef_[0]
intercept = lr.intercept_

# Bounded calibration → improves private score stability
slope = np.clip(slope, 0.92, 1.08)
intercept = np.clip(intercept, -2.5, 2.5)

calibrated_oof = slope * oof_pred + intercept
calibrated_rmse = mean_squared_error(y, calibrated_oof, squared=False)
print(f"Calibrated OOF RMSE: {calibrated_rmse:.5f}")
print(f"Slope={slope:.4f}, Intercept={intercept:.4f}")

# ---------------- FINAL PREDICTION ----------------
base_test_pred = test_pred_folds.mean(axis=0)
final_pred = slope * base_test_pred + intercept

print("\nFinal Test Prediction Stats:")
print(pd.Series(final_pred).describe())

submission = pd.DataFrame({
    "id": test_ids,
    "BeatsPerMinute": final_pred
})

submission.to_csv("submission_v10_pure.csv", index=False)
print("Saved submission_v10_pure.csv")



# ============================================================
#  Kaggle Playground - Predict BeatsPerMinute
#  Version 9: Single LGBM + Original Data + Linear Calibration
#  Improvement: 
#    - Uses Original Dataset for better generalization
#    - Replaces hardcoded variance scaling with Linear Regression Calibration on OOF
#    - Adds integer-check features
# ============================================================

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# ---------------- CONFIG ----------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "/kaggle/input/playground-series-s5e9"
ORIG_DATA_DIR = "/kaggle/input/bpm-prediction-challenge" # Common path for original data
TARGET_COL = "BeatsPerMinute"

# ---------------- LOAD DATA ----------------
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# Try to load original data
try:
    orig_train = pd.read_csv(os.path.join(ORIG_DATA_DIR, "Train.csv"))
    # Ensure columns match
    common_cols = [c for c in train.columns if c in orig_train.columns]
    orig_train = orig_train[common_cols]
    
    # Add source flag
    train['is_generated'] = 1
    test['is_generated'] = 1
    orig_train['is_generated'] = 0
    
    # Concatenate
    train = pd.concat([train, orig_train], axis=0).reset_index(drop=True)
    print(f"Original data added. New train shape: {train.shape}")
except FileNotFoundError:
    print("Original dataset not found. Proceeding with synthetic only.")
    train['is_generated'] = 1
    test['is_generated'] = 1

y = train[TARGET_COL]
X = train.drop(columns=[TARGET_COL, "id"]) # Drop ID
X_test = test.drop(columns=["id"])
test_ids = test["id"]

# ---------------- FEATURE ENGINEERING ----------------
# Basic cleaning + simple synthetic-specific features
for df in [X, X_test]:
    # Encode categoricals
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
            # Simple frequency encoding can be better than one-hot for trees
            freq_enc = df[col].value_counts().to_dict()
            df[col] = df[col].map(freq_enc)
        else:
            df[col] = df[col].fillna(df[col].median())
            
    # Add Integer Check Feature (Synthetic data often has float artifacts)
    # Checking if numeric columns are close to integers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'is_generated':
            # Feature: fractional part
            df[f'{col}_frac'] = df[col] % 1
            # Feature: is integer?
            df[f'{col}_is_int'] = (df[col] % 1 == 0).astype(int)

# Align columns
X_test = X_test.reindex(columns=X.columns, fill_value=0)
print("After Feature Engineering:", X.shape, X_test.shape)

X_vals = X.values
X_test_vals = X_test.values

# ---------------- MODEL PARAMS ----------------
# Slightly adjusted params for larger dataset (if original added)
lgb_params = {
    "n_estimators": 3000,
    "learning_rate": 0.009, # Lower LR for better convergence
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_samples": 100,
    "reg_alpha": 0.6,      # Increased reg
    "reg_lambda": 0.5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "metric": "rmse",
    "verbosity": -1
}

# ---------------- K-FOLD TRAIN ----------------
N_SPLITS = 14 # Increase folds for better OOF calibration
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_pred = np.zeros(X_vals.shape[0])
test_pred_folds = np.zeros((N_SPLITS, X_test_vals.shape[0]))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
    X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(0)
        ]
    )

    va_pred = model.predict(X_va)
    oof_pred[va_idx] = va_pred
    test_pred_folds[fold] = model.predict(X_test_vals)
    
    rmse = mean_squared_error(y_va, va_pred, squared=False)
    print(f"Fold {fold+1} RMSE: {rmse:.5f}")

# ---------------- POST-PROCESSING: LINEAR CALIBRATION ----------------
# Instead of hardcoding target_std = 0.6, we learn the optimal scaling from OOF
# We fit a Linear Regression: True_Target ~ a * OOF_Pred + b
# This automatically finds the best mean shift (b) and variance shrinkage (a)

print("\n--- Calibration ---")
raw_oof_rmse = mean_squared_error(y, oof_pred, squared=False)
print(f"Raw OOF RMSE: {raw_oof_rmse:.5f}")

lr = LinearRegression()
lr.fit(oof_pred.reshape(-1, 1), y)

print(f"Calibration Slope (Shrinkage): {lr.coef_[0]:.4f}")
print(f"Calibration Intercept (Mean Shift): {lr.intercept_:.4f}")

# Apply calibration to OOF to check improvement
calibrated_oof = lr.predict(oof_pred.reshape(-1, 1))
calibrated_rmse = mean_squared_error(y, calibrated_oof, squared=False)
print(f"Calibrated OOF RMSE: {calibrated_rmse:.5f}")

# ---------------- FINAL PREDICTION ----------------
# Average raw predictions from folds
base_test_pred = test_pred_folds.mean(axis=0)

# Apply the learned calibration to test predictions
final_pred = lr.predict(base_test_pred.reshape(-1, 1))

print("\nFinal Test Prediction Stats:")
print(pd.Series(final_pred).describe())

# ---------------- SAVE ----------------
submission = pd.DataFrame({
    "id": test_ids,
    "BeatsPerMinute": final_pred
})

submission.to_csv("submission_calibrated.csv", index=False)
print("\nSaved submission_calibrated.csv")


# ============================================================
#  Kaggle Playground S5E9 - Predict BeatsPerMinute
#  Version 10: Optimized Params + Targeted Calibration
#  Target RMSE: ~26.404
# ============================================================

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "/kaggle/input/playground-series-s5e9"
ORIG_DATA_DIR = "/kaggle/input/bpm-prediction-challenge" 
TARGET_COL = "BeatsPerMinute"
N_SPLITS = 15  # Tăng lên 15 để OOF ổn định hơn

# ---------------- LOAD DATA ----------------
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# Xử lý dữ liệu gốc (Original Data)
try:
    orig_train = pd.read_csv(os.path.join(ORIG_DATA_DIR, "Train.csv"))
    common_cols = [c for c in train.columns if c in orig_train.columns]
    orig_train = orig_train[common_cols]
    
    # Đánh dấu nguồn dữ liệu
    train['is_generated'] = 1
    test['is_generated'] = 1
    orig_train['is_generated'] = 0
    
    # Gộp dữ liệu
    train = pd.concat([train, orig_train], axis=0).reset_index(drop=True)
    print(f"Original data added. New train shape: {train.shape}")
except FileNotFoundError:
    print("Original dataset not found. Proceeding with synthetic only.")
    train['is_generated'] = 1
    test['is_generated'] = 1

y = train[TARGET_COL]
X = train.drop(columns=[TARGET_COL, "id"]) 
X_test = test.drop(columns=["id"])
test_ids = test["id"]

# ---------------- FEATURE ENGINEERING ----------------
numeric_cols = X.select_dtypes(include=[np.number]).columns

for df in [X, X_test]:
    # 1. Xử lý Categorical
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
            freq_enc = df[col].value_counts().to_dict()
            df[col] = df[col].map(freq_enc)
        else:
            # Fill NA cho numeric bằng median
            df[col] = df[col].fillna(df[col].median())
    
    # 2. Integer Check Features (Quan trọng cho Synthetic Data)
    # Giúp model phát hiện các mẫu "lỗi" thập phân do quá trình sinh dữ liệu
    for col in numeric_cols:
        if col != 'is_generated':
            # Phần thập phân
            df[f'{col}_frac'] = df[col] % 1
            # Là số nguyên hay không?
            df[f'{col}_is_int'] = (df[col] % 1 == 0).astype(int)

# Align columns (đảm bảo thứ tự cột giống nhau)
X_test = X_test.reindex(columns=X.columns, fill_value=0)
print("After Feature Engineering:", X.shape, X_test.shape)

X_vals = X.values
X_test_vals = X_test.values

# ---------------- MODEL PARAMS (TUNED) ----------------
# Đã tinh chỉnh để giảm Variance (giảm RMSE nhẹ)
lgb_params = {
    "n_estimators": 5000,       # Tăng thêm cây để bù cho learning rate thấp
    "learning_rate": 0.007,     # Giảm LR thêm chút nữa để hội tụ mượt
    "num_leaves": 63,           # TĂNG GẤP ĐÔI (Quan trọng nhất để giảm RMSE)
    "max_depth": -1,
    "subsample": 0.6,           # Giảm subsample để tăng tính ngẫu nhiên
    "colsample_bytree": 0.5,    # Giảm colsample để chống Overfit khi tăng num_leaves
    "min_child_samples": 40,    # GIẢM MẠNH (từ 128 -> 40) để bắt chi tiết nhỏ
    "reg_alpha": 0.1,           # Giảm Regularization để model "bung" sức mạnh
    "reg_lambda": 0.1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "metric": "rmse",
    "verbosity": -1
}

# ---------------- K-FOLD TRAIN ----------------
# Giữ nguyên N_SPLITS = 15
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_pred = np.zeros(X_vals.shape[0])
test_pred_folds = np.zeros((N_SPLITS, X_test_vals.shape[0]))

print(f"\nStarting Training V11 (Aggressive Params)...")

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
    X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=300, verbose=False), # Tăng patience
            lgb.log_evaluation(0)
        ]
    )

    va_pred = model.predict(X_va)
    oof_pred[va_idx] = va_pred
    test_pred_folds[fold] = model.predict(X_test_vals)
    
    # In ra mỗi fold để theo dõi tiến độ
    rmse = mean_squared_error(y_va, va_pred, squared=False)
    if (fold+1) % 5 == 0:
        print(f"Fold {fold+1} RMSE: {rmse:.5f}")


# ---------------- REVISED CALIBRATION ----------------
# Quay lại Calibrate trên TOÀN BỘ dữ liệu (cả gốc + nhân tạo)
# Lý do: Với model phức tạp hơn (63 leaves), ta cần nhiều điểm dữ liệu hơn để fit đường hồi quy chuẩn
print("\n--- Calibration (All Data) ---")
lr = LinearRegression()
lr.fit(oof_pred.reshape(-1, 1), y) # Fit trên toàn bộ y

print(f"Calibration Slope: {lr.coef_[0]:.5f} (Mục tiêu: Gần 1.0 hơn so với 1.16)")
print(f"Calibration Intercept: {lr.intercept_:.5f}")

base_test_pred = test_pred_folds.mean(axis=0)
final_pred = lr.predict(base_test_pred.reshape(-1, 1))

submission = pd.DataFrame({"id": test_ids, "BeatsPerMinute": final_pred})
submission.to_csv("submission_v11.csv", index=False)
print("\nSaved submission_v11.csv")


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

# 1. Tải dữ liệu
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
test_ids = test['id'].copy()

# 2. Tách biến mục tiêu và loại bỏ cột không dùng (id)
y = train['BeatsPerMinute']
train = train.drop(columns=['id','BeatsPerMinute'])
X = train.copy()
X_test = test.drop(columns=['id']).copy()

# 3. Xử lý dữ liệu: mã hóa categorical và fillna
for col in X.columns:
    if X[col].dtype == 'object':
        # Encode giá trị đếm (frequency) cho biến phân loại
        X[col] = X[col].fillna('missing')
        freq = X[col].value_counts().to_dict()
        X[col] = X[col].map(freq)
        # Áp dụng cùng mã hóa cho X_test (những giá trị mới sẽ bị NaN -> gán 0)
        X_test[col] = X_test[col].fillna('missing')
        X_test[col] = X_test[col].map(freq).fillna(0)
# Điền median cho các cột số
num_cols = X.select_dtypes(include=[np.number]).columns
for col in num_cols:
    med = X[col].median()
    X[col].fillna(med, inplace=True)
    X_test[col].fillna(med, inplace=True)

# 4. Tạo đặc trưng tổng quát thêm: phần dư (fractional) và cờ kiểm tra số nguyên
for col in num_cols:
    X[f'{col}_frac'] = X[col] % 1
    X_test[f'{col}_frac'] = X_test[col] % 1
    X[f'{col}_is_int'] = (X[f'{col}_frac'] == 0).astype(int)
    X_test[f'{col}_is_int'] = (X_test[f'{col}_frac'] == 0).astype(int)

# 5. Đảm bảo số cột giống nhau giữa X và X_test
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# 6. Chuẩn bị dữ liệu cho huấn luyện
X_vals = X.values
X_test_vals = X_test.values

# 7. Thiết lập tham số và K-Fold
lgb_params = {
    "objective": "regression",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.6,
    "reg_lambda": 0.5,
    "random_state": 42,
    "n_jobs": -1
}
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_pred = np.zeros(X_vals.shape[0])
test_pred = np.zeros(X_test_vals.shape[0])

# 8. Huấn luyện LGBM với Cross-Validation
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
    X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        early_stopping_rounds=100,
        verbose=False
    )
    # Dự đoán và tính RMSE trên tập validation
    va_pred = model.predict(X_va)
    oof_pred[va_idx] = va_pred
    rmse = mean_squared_error(y_va, va_pred, squared=False)
    print(f"Fold {fold+1} RMSE: {rmse:.5f}")
    # Tích lũy dự đoán trên tập test
    test_pred += model.predict(X_test_vals)

# Trung bình dự đoán trên các fold
test_pred /= N_SPLITS

# 9. Hiệu chỉnh đầu ra (Linear Calibration)
print("\n--- Calibration ---")
raw_oof_rmse = mean_squared_error(y, oof_pred, squared=False)
print(f"Raw OOF RMSE: {raw_oof_rmse:.5f}")
lr = LinearRegression().fit(oof_pred.reshape(-1, 1), y)
slope = lr.coef_[0]
intercept = lr.intercept_
print(f"Calibration Slope: {slope:.3f}, Intercept: {intercept:.3f}")
# Áp dụng hiệu chỉnh lên OOF để xem RMSE cải thiện
cal_oof = lr.predict(oof_pred.reshape(-1, 1))
calibrated_rmse = mean_squared_error(y, cal_oof, squared=False)
print(f"Calibrated OOF RMSE: {calibrated_rmse:.5f}")

# 10. Dự đoán cuối cùng trên tập test với hiệu chỉnh
final_pred = lr.predict(test_pred.reshape(-1, 1))

# 11. Lưu kết quả submit
submission = pd.DataFrame({"id": test_ids, "BeatsPerMinute": final_pred})
submission.to_csv("submission.csv", index=False)
print("\nFinal submission saved. Prediction stats:")
print(pd.Series(final_pred).describe().apply(lambda x: f"{x:,.2f}"))


# ============================================================
#  Kaggle Playground S5E9 - Version 12 (Full & Fixed)
#  Strategy: Slow Learning + Ridge Calibration
#  Fix: Ensures 'is_generated' column exists
# ============================================================

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import warnings

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "/kaggle/input/playground-series-s5e9"
ORIG_DATA_DIR = "/kaggle/input/bpm-prediction-challenge"
TARGET_COL = "BeatsPerMinute"
N_SPLITS = 15 

# ---------------- LOAD DATA & CREATE 'is_generated' ----------------
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# Xử lý dữ liệu gốc (Original Data)
try:
    orig_train = pd.read_csv(os.path.join(ORIG_DATA_DIR, "Train.csv"))
    # Chỉ lấy các cột chung
    common_cols = [c for c in train.columns if c in orig_train.columns]
    orig_train = orig_train[common_cols]
    
    # Đánh dấu nguồn dữ liệu (QUAN TRỌNG ĐỂ KHÔNG BỊ LỖI KEYERROR)
    train['is_generated'] = 1
    test['is_generated'] = 1
    orig_train['is_generated'] = 0
    
    # Gộp dữ liệu
    train = pd.concat([train, orig_train], axis=0).reset_index(drop=True)
    print(f"Original data added. New train shape: {train.shape}")
except FileNotFoundError:
    print("Original dataset not found. Proceeding with synthetic only.")
    train['is_generated'] = 1
    test['is_generated'] = 1

y = train[TARGET_COL]
# Lưu ý: Không drop 'is_generated' khỏi train gốc, chỉ drop khi tạo X
X = train.drop(columns=[TARGET_COL, "id"]) 
X_test = test.drop(columns=["id"])
test_ids = test["id"]

# ---------------- FEATURE ENGINEERING ----------------
numeric_cols = X.select_dtypes(include=[np.number]).columns

for df in [X, X_test]:
    # 1. Xử lý Categorical
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
            freq_enc = df[col].value_counts().to_dict()
            df[col] = df[col].map(freq_enc)
        else:
            df[col] = df[col].fillna(df[col].median())
    
    # 2. Integer Check Features
    for col in numeric_cols:
        if col != 'is_generated':
            df[f'{col}_frac'] = df[col] % 1
            df[f'{col}_is_int'] = (df[col] % 1 == 0).astype(int)

# Align columns
X_test = X_test.reindex(columns=X.columns, fill_value=0)
print("After Feature Engineering:", X.shape, X_test.shape)

X_vals = X.values
X_test_vals = X_test.values

# ---------------- MODEL PARAMS (VERSION 12 - STABLE) ----------------
# Cấu hình "Slow Cook": Học chậm, nhiều cây, Ridge Calibration
lgb_params = {
    "n_estimators": 7000,       
    "learning_rate": 0.009,     # Học rất chậm
    "num_leaves": 34,           # Vừa đủ
    "max_depth": -1,
    "subsample": 0.75,          
    "colsample_bytree": 0.55,   
    "min_child_samples": 100,   
    "reg_alpha": 0.6,           
    "reg_lambda": 0.5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "metric": "rmse",
    "verbosity": -1
}

# ---------------- K-FOLD TRAIN ----------------
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_pred = np.zeros(X_vals.shape[0])
test_pred_folds = np.zeros((N_SPLITS, X_test_vals.shape[0]))

print(f"\nStarting Training V12 (Slow Cook)...")

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
    X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=400, verbose=False),
            lgb.log_evaluation(0)
        ]
    )

    va_pred = model.predict(X_va)
    oof_pred[va_idx] = va_pred
    test_pred_folds[fold] = model.predict(X_test_vals)
    
    if (fold+1) % 5 == 0:
        rmse = mean_squared_error(y_va, va_pred, squared=False)
        print(f"Fold {fold+1} RMSE: {rmse:.5f}")

# ---------------- RIDGE CALIBRATION (FIXED) ----------------
print("\n--- Calibration (Ridge on Synthetic) ---")

# Fix lỗi KeyError: Bây giờ biến train chắc chắn đã có 'is_generated'
mask_generated = train['is_generated'] == 1

# Sử dụng Ridge Regression để ổn định hóa kết quả
ridge = Ridge(alpha=10.0) 
ridge.fit(oof_pred[mask_generated].reshape(-1, 1), y[mask_generated])

print(f"Ridge Slope: {ridge.coef_[0]:.5f}")
print(f"Ridge Intercept: {ridge.intercept_:.5f}")

calibrated_oof = ridge.predict(oof_pred.reshape(-1, 1))
calibrated_rmse = mean_squared_error(y, calibrated_oof, squared=False)
print(f"Calibrated Global RMSE: {calibrated_rmse:.5f}")

# ---------------- PREDICTION ----------------
base_test_pred = test_pred_folds.mean(axis=0)
final_pred = ridge.predict(base_test_pred.reshape(-1, 1))

submission = pd.DataFrame({"id": test_ids, "BeatsPerMinute": final_pred})
submission.to_csv("submission_v12_ridge.csv", index=False)
print("\nSaved submission_v12_ridge.csv successfully!")


# ============================================================
#  Kaggle Playground S5E9 - Version 14: The Super Baseline
#  Strategy: Single LGBM + Seed Averaging + Ridge Calibration
#  Philosophy: "Simple is Better"
# ============================================================

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import warnings

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
# Thay vì 1 seed, ta dùng 3 seed để lấy trung bình (giảm Variance cực tốt)
SEEDS = [42, 2024, 777] 
N_SPLITS = 10 

DATA_DIR = "/kaggle/input/playground-series-s5e9"
ORIG_DATA_DIR = "/kaggle/input/bpm-prediction-challenge" 
TARGET_COL = "BeatsPerMinute"

# ---------------- LOAD DATA & PREP ----------------
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

try:
    orig_train = pd.read_csv(os.path.join(ORIG_DATA_DIR, "Train.csv"))
    common_cols = [c for c in train.columns if c in orig_train.columns]
    orig_train = orig_train[common_cols]
    
    train['is_generated'] = 1
    test['is_generated'] = 1
    orig_train['is_generated'] = 0
    
    train = pd.concat([train, orig_train], axis=0).reset_index(drop=True)
    print(f"Data Loaded with Original. Shape: {train.shape}")
except:
    print("Original data not found. Using synthetic only.")
    train['is_generated'] = 1
    test['is_generated'] = 1

y = train[TARGET_COL]
X = train.drop(columns=[TARGET_COL, "id"]) 
X_test = test.drop(columns=["id"])
test_ids = test["id"]

# Feature Engineering (Giữ nguyên phần xử lý số nguyên hiệu quả)
numeric_cols = X.select_dtypes(include=[np.number]).columns
for df in [X, X_test]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
            freq_enc = df[col].value_counts().to_dict()
            df[col] = df[col].map(freq_enc)
        else:
            df[col] = df[col].fillna(df[col].median())
    
    for col in numeric_cols:
        if col != 'is_generated':
            df[f'{col}_frac'] = df[col] % 1
            df[f'{col}_is_int'] = (df[col] % 1 == 0).astype(int)

X_test = X_test.reindex(columns=X.columns, fill_value=0)
X_vals = X.values
X_test_vals = X_test.values

# ---------------- MODEL PARAMS ----------------
# Params V12 (Slow Cook) đã được chứng minh là tốt nhất
base_params = {
    "n_estimators": 3000,       
    "learning_rate": 0.009,     
    "num_leaves": 34,           
    "max_depth": -1,
    "subsample": 0.75,          
    "colsample_bytree": 0.55,   
    "min_child_samples": 100,   
    "reg_alpha": 0.6,           
    "reg_lambda": 0.5,
    "n_jobs": -1,
    "metric": "rmse",
    "verbosity": -1
}

# ---------------- SEED AVERAGING LOOP ----------------
# Lưu trữ kết quả của từng Seed
oof_preds_total = np.zeros(X_vals.shape[0])
test_preds_total = np.zeros(X_test_vals.shape[0])

print(f"--- Starting Seed Averaging (Seeds: {SEEDS}) ---")

for i, seed in enumerate(SEEDS):
    print(f"\nTraining Seed {seed} ({i+1}/{len(SEEDS)})...")
    
    # Cập nhật seed cho params và KFold
    params = base_params.copy()
    params['random_state'] = seed
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    
    oof_seed = np.zeros(X_vals.shape[0])
    test_seed = np.zeros(X_test_vals.shape[0])
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
        X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(300, verbose=False)]
        )
        
        oof_seed[va_idx] = model.predict(X_va)
        test_seed += model.predict(X_test_vals) / N_SPLITS
    
    # Cộng dồn kết quả
    oof_preds_total += oof_seed / len(SEEDS)
    test_preds_total += test_seed / len(SEEDS)
    
    rmse_seed = mean_squared_error(y, oof_seed, squared=False)
    print(f"Seed {seed} Raw RMSE: {rmse_seed:.5f}")

print("\n--- All Seeds Completed ---")
raw_oof_rmse = mean_squared_error(y, oof_preds_total, squared=False)
print(f"Combined Raw OOF RMSE: {raw_oof_rmse:.5f}")

# ---------------- RIDGE CALIBRATION ----------------
print("\n--- Final Calibration (Ridge) ---")

mask_generated = train['is_generated'] == 1

# Calibrate trên kết quả trung bình của 3 seeds
ridge = Ridge(alpha=10.0)
ridge.fit(oof_preds_total[mask_generated].reshape(-1, 1), y[mask_generated])

print(f"Ridge Slope: {ridge.coef_[0]:.5f}")
print(f"Ridge Intercept: {ridge.intercept_:.5f}")

calibrated_oof = ridge.predict(oof_preds_total.reshape(-1, 1))
calibrated_rmse = mean_squared_error(y, calibrated_oof, squared=False)
print(f"Final Calibrated RMSE: {calibrated_rmse:.5f}")

# ---------------- SUBMISSION ----------------
final_pred = ridge.predict(test_preds_total.reshape(-1, 1))

submission = pd.DataFrame({"id": test_ids, "BeatsPerMinute": final_pred})
submission.to_csv("submission_v14_baseline_avg.csv", index=False)
print("\nSaved submission_v14_baseline_avg.csv")


# ============================================================
#  Kaggle Playground S5E9 - Version 15: Geometric Baseline
#  Strategy: Parameter Diversity + Geometric Mean Ensemble
#  Score Target: < 26.40469
# ============================================================

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import warnings

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
N_SPLITS = 12 # Tăng nhẹ số fold để ổn định hơn
DATA_DIR = "/kaggle/input/playground-series-s5e9"
ORIG_DATA_DIR = "/kaggle/input/bpm-prediction-challenge" 
TARGET_COL = "BeatsPerMinute"

# Cấu hình 3 biến thể model để tạo Diversity (Sự đa dạng)
# Thay vì chỉ đổi seed, ta đổi cả độ phức tạp của model
MODEL_VARIANTS = [
    {"seed": 42,   "leaves": 31, "lr": 0.006, "name": "Conservative"},
    {"seed": 2024, "leaves": 34, "lr": 0.006, "name": "Balanced"},
    {"seed": 777,  "leaves": 37, "lr": 0.0055, "name": "Aggressive"} # LR thấp hơn do model phức tạp hơn
]

# ---------------- LOAD DATA ----------------
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

try:
    orig_train = pd.read_csv(os.path.join(ORIG_DATA_DIR, "Train.csv"))
    common_cols = [c for c in train.columns if c in orig_train.columns]
    orig_train = orig_train[common_cols]
    
    train['is_generated'] = 1
    test['is_generated'] = 1
    orig_train['is_generated'] = 0
    
    train = pd.concat([train, orig_train], axis=0).reset_index(drop=True)
    print(f"Data Loaded. Shape: {train.shape}")
except:
    train['is_generated'] = 1
    test['is_generated'] = 1

y = train[TARGET_COL]
X = train.drop(columns=[TARGET_COL, "id"]) 
X_test = test.drop(columns=["id"])
test_ids = test["id"]

# ---------------- FEATURE ENGINEERING ----------------
numeric_cols = X.select_dtypes(include=[np.number]).columns
for df in [X, X_test]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
            freq_enc = df[col].value_counts().to_dict()
            df[col] = df[col].map(freq_enc)
        else:
            df[col] = df[col].fillna(df[col].median())
    
    for col in numeric_cols:
        if col != 'is_generated':
            df[f'{col}_frac'] = df[col] % 1
            df[f'{col}_is_int'] = (df[col] % 1 == 0).astype(int)

X_test = X_test.reindex(columns=X.columns, fill_value=0)
X_vals = X.values
X_test_vals = X_test.values

# ---------------- TRAINING LOOP WITH DIVERSITY ----------------
# Lưu kết quả của từng biến thể để Blending sau
oof_preds_dict = {}
test_preds_dict = {}

base_params = {
    "n_estimators": 6000,       
    "max_depth": -1,
    "subsample": 0.75,          
    "colsample_bytree": 0.55,   
    "min_child_samples": 100,   
    "reg_alpha": 0.5,           
    "reg_lambda": 0.5,
    "n_jobs": -1,
    "metric": "rmse",
    "verbosity": -1
}

print(f"--- Starting Geometric Ensemble Training ---")

for variant in MODEL_VARIANTS:
    v_name = variant["name"]
    print(f"\nTraining Variant: {v_name} (Leaves={variant['leaves']})...")
    
    # Update params
    params = base_params.copy()
    params['random_state'] = variant['seed']
    params['num_leaves'] = variant['leaves']
    params['learning_rate'] = variant['lr']
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=variant['seed'])
    
    oof_seed = np.zeros(X_vals.shape[0])
    test_seed = np.zeros(X_test_vals.shape[0])
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
        X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(300, verbose=False)]
        )
        
        oof_seed[va_idx] = model.predict(X_va)
        test_seed += model.predict(X_test_vals) / N_SPLITS
    
    rmse_seed = mean_squared_error(y, oof_seed, squared=False)
    print(f"{v_name} Raw RMSE: {rmse_seed:.5f}")
    
    # Lưu lại để blend
    oof_preds_dict[v_name] = oof_seed
    test_preds_dict[v_name] = test_seed

# ---------------- GEOMETRIC MEAN BLENDING ----------------
print("\n--- Applying Geometric Mean Blending ---")

# Gom tất cả dự đoán vào 1 mảng
all_oof = np.column_stack(list(oof_preds_dict.values()))
all_test = np.column_stack(list(test_preds_dict.values()))

# Công thức Geometric Mean: exp(mean(log(x)))
# Lưu ý: BPM luôn dương nên log an toàn. Nếu có <=0 cần xử lý, nhưng BPM min > 30.
final_oof_geo = np.exp(np.mean(np.log(all_oof), axis=1))
final_test_geo = np.exp(np.mean(np.log(all_test), axis=1))

geo_rmse = mean_squared_error(y, final_oof_geo, squared=False)
print(f"Geometric Mean OOF RMSE: {geo_rmse:.5f}")

# ---------------- RIDGE CALIBRATION ----------------
print("\n--- Final Calibration (Ridge on GeoMean) ---")

mask_generated = train['is_generated'] == 1

ridge = Ridge(alpha=10.0)
ridge.fit(final_oof_geo[mask_generated].reshape(-1, 1), y[mask_generated])

print(f"Ridge Slope: {ridge.coef_[0]:.5f}")
print(f"Ridge Intercept: {ridge.intercept_:.5f}")

calibrated_oof = ridge.predict(final_oof_geo.reshape(-1, 1))
calibrated_rmse = mean_squared_error(y, calibrated_oof, squared=False)
print(f"Calibrated Final RMSE: {calibrated_rmse:.5f}")

# ---------------- SUBMISSION ----------------
final_pred = ridge.predict(final_test_geo.reshape(-1, 1))

submission = pd.DataFrame({"id": test_ids, "BeatsPerMinute": final_pred})
submission.to_csv("submission_v15_geo_blend.csv", index=False)
print("\nSaved submission_v15_geo_blend.csv")


# ============================================================
#  Kaggle Playground S5E9 - Version 15: Diverse Baseline
#  Strategy: Diverse Param Seeds + Geometric Mean + Soft Rounding
#  Target: Beat 26.40467
# ============================================================

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import warnings

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
N_SPLITS = 12  # Increased slightly for stability
DATA_DIR = "/kaggle/input/playground-series-s5e9"
ORIG_DATA_DIR = "/kaggle/input/bpm-prediction-challenge" 
TARGET_COL = "BeatsPerMinute"

# DIVERSITY STRATEGY:
# Instead of identical params, we create 3 variants.
# 1. Conservative: Fewer leaves, learns broader patterns.
# 2. Balanced: Your current best settings.
# 3. Aggressive: More leaves, learns finer details.
MODEL_VARIANTS = [
    {"seed": 42,   "leaves": 31, "lr": 0.009,  "name": "Conservative"},
    {"seed": 2024, "leaves": 34, "lr": 0.009,  "name": "Balanced (Ref)"},
    {"seed": 777,  "leaves": 38, "lr": 0.0085, "name": "Aggressive"} 
]

# ---------------- LOAD DATA ----------------
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

try:
    orig_train = pd.read_csv(os.path.join(ORIG_DATA_DIR, "Train.csv"))
    common_cols = [c for c in train.columns if c in orig_train.columns]
    orig_train = orig_train[common_cols]
    
    train['is_generated'] = 1
    test['is_generated'] = 1
    orig_train['is_generated'] = 0
    
    train = pd.concat([train, orig_train], axis=0).reset_index(drop=True)
    print(f"Data Loaded with Original. Shape: {train.shape}")
except:
    train['is_generated'] = 1
    test['is_generated'] = 1

y = train[TARGET_COL]
X = train.drop(columns=[TARGET_COL, "id"]) 
X_test = test.drop(columns=["id"])
test_ids = test["id"]

# ---------------- FEATURE ENGINEERING ----------------
numeric_cols = X.select_dtypes(include=[np.number]).columns
for df in [X, X_test]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
            freq_enc = df[col].value_counts().to_dict()
            df[col] = df[col].map(freq_enc)
        else:
            df[col] = df[col].fillna(df[col].median())
    
    for col in numeric_cols:
        if col != 'is_generated':
            df[f'{col}_frac'] = df[col] % 1
            df[f'{col}_is_int'] = (df[col] % 1 == 0).astype(int)

X_test = X_test.reindex(columns=X.columns, fill_value=0)
X_vals = X.values
X_test_vals = X_test.values

# ---------------- TRAINING LOOP ----------------
# We store predictions in lists to apply Geometric Mean later
oof_preds_list = []
test_preds_list = []

# Base Params (Shared)
base_params = {
    "n_estimators": 3500, # Increased slightly to ensure convergence
    "max_depth": -1,
    "subsample": 0.75,          
    "colsample_bytree": 0.55,   
    "min_child_samples": 100,   
    "reg_alpha": 0.6,           
    "reg_lambda": 0.5,
    "n_jobs": -1,
    "metric": "rmse",
    "verbosity": -1
}

print(f"--- Starting Diverse Training ---")

for variant in MODEL_VARIANTS:
    v_name = variant['name']
    print(f"\nTraining {v_name} (Leaves={variant['leaves']})...")
    
    # Specific Params
    params = base_params.copy()
    params['random_state'] = variant['seed']
    params['num_leaves'] = variant['leaves']
    params['learning_rate'] = variant['lr']
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=variant['seed'])
    
    oof_seed = np.zeros(X_vals.shape[0])
    test_seed = np.zeros(X_test_vals.shape[0])
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
        X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(300, verbose=False)]
        )
        
        oof_seed[va_idx] = model.predict(X_va)
        test_seed += model.predict(X_test_vals) / N_SPLITS
    
    rmse_seed = mean_squared_error(y, oof_seed, squared=False)
    print(f"{v_name} RMSE: {rmse_seed:.5f}")
    
    oof_preds_list.append(oof_seed)
    test_preds_list.append(test_seed)

# ---------------- GEOMETRIC MEAN AGGREGATION ----------------
print("\n--- Aggregating with Geometric Mean ---")

# Stack predictions
oof_stack = np.column_stack(oof_preds_list)
test_stack = np.column_stack(test_preds_list)

# Geometric Mean Formula: exp(mean(log(x)))
# This handles outliers better than Arithmetic Mean
geo_oof = np.exp(np.mean(np.log(oof_stack), axis=1))
geo_test = np.exp(np.mean(np.log(test_stack), axis=1))

raw_geo_rmse = mean_squared_error(y, geo_oof, squared=False)
print(f"Geometric Mean OOF RMSE: {raw_geo_rmse:.5f}")

# ---------------- RIDGE CALIBRATION ----------------
print("\n--- Ridge Calibration ---")
mask_generated = train['is_generated'] == 1

ridge = Ridge(alpha=10.0)
ridge.fit(geo_oof[mask_generated].reshape(-1, 1), y[mask_generated])

print(f"Ridge Slope: {ridge.coef_[0]:.5f}")
calibrated_oof = ridge.predict(geo_oof.reshape(-1, 1))
calibrated_test = ridge.predict(geo_test.reshape(-1, 1))

calibrated_rmse = mean_squared_error(y, calibrated_oof, squared=False)
print(f"Calibrated RMSE: {calibrated_rmse:.5f}")

# ---------------- SOFT INTEGER BLENDING (The Finisher) ----------------
print("\n--- Soft Integer Blending Optimization ---")
# Since BPM is integer, mixing in a small % of rounded values often helps.
# We find the optimal 'ratio' using the OOF data.

best_ratio = 0.0
best_rmse = calibrated_rmse

# Search grid: 0% to 20%
for r in np.linspace(0, 0.2, 50):
    # Blend: (1-r)*Float + r*Rounded
    temp_pred = (1 - r) * calibrated_oof + r * np.round(calibrated_oof)
    temp_rmse = mean_squared_error(y, temp_pred, squared=False)
    
    if temp_rmse < best_rmse:
        best_rmse = temp_rmse
        best_ratio = r

print(f"Best Rounding Ratio: {best_ratio:.4f}")
print(f"Final Optimized OOF RMSE: {best_rmse:.5f}")

# Apply best ratio to Test
final_pred = (1 - best_ratio) * calibrated_test + best_ratio * np.round(calibrated_test)

# ---------------- SUBMISSION ----------------
submission = pd.DataFrame({"id": test_ids, "BeatsPerMinute": final_pred})
submission.to_csv("submission_v15_diverse_geo.csv", index=False)
print("\nSaved submission_v15_diverse_geo.csv")

