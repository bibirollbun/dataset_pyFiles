# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


SEED = 42
N_SPLITS = 5

TARGET = "diagnosed_diabetes"
ID_COL = "id"

pd.set_option("display.max_columns", 200)
np.random.seed(SEED)



train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)



display(train.head(3))
display(test.head(3))


print(train.info())
print("\nTarget distribution:")
print(train[TARGET].value_counts())
print(train[TARGET].value_counts(normalize=True))

sns.countplot(x=TARGET, data=train)
plt.title("Target distribution")
plt.show()


missing_counts = train.isnull().sum().sort_values(ascending=False)
print("Missing values (top 20):")
print(missing_counts.head(20))


def safe_div(a, b, eps=1e-6):
    return a / (b + eps)

def enhance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

# --- Metabolic ratios ---
    if {"triglycerides", "hdl_cholesterol"}.issubset(df.columns):
        df["tg_hdl_ratio"] = safe_div(df["triglycerides"], df["hdl_cholesterol"])

    if {"ldl_cholesterol", "hdl_cholesterol"}.issubset(df.columns):
        df["ldl_hdl_ratio"] = safe_div(df["ldl_cholesterol"], df["hdl_cholesterol"])

    if {"cholesterol_total", "hdl_cholesterol"}.issubset(df.columns):
        df["non_hdl"] = df["cholesterol_total"] - df["hdl_cholesterol"]

# --- Hemodynamics ---
    if {"systolic_bp", "diastolic_bp"}.issubset(df.columns):
        df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
        df["map"] = df["diastolic_bp"] + (df["pulse_pressure"] / 3.0)

# --- Body composition interactions ---
    if {"bmi", "waist_to_hip_ratio"}.issubset(df.columns):
        df["bmi_x_whr"] = df["bmi"] * df["waist_to_hip_ratio"]

# --- Lifestyle interactions ---
    if {"screen_time_hours_per_day", "physical_activity_minutes_per_week"}.issubset(df.columns):
    # 'sedentary pressure' proxy: more screen + less activity => higher risk
        df["screen_minus_activity"] = df["screen_time_hours_per_day"] - (df["physical_activity_minutes_per_week"] / 200.0)
    
# --- Simple risk flags (soft, not too many) ---
    if "bmi" in df.columns:
        df["bmi_obese"] = (df["bmi"] >= 30).astype(np.int8)
        df["bmi_overweight"] = ((df["bmi"] >= 25) & (df["bmi"] < 30)).astype(np.int8)

    if "age" in df.columns:
        df["age_50_plus"] = (df["age"] >= 50).astype(np.int8)
        df["age_65_plus"] = (df["age"] >= 65).astype(np.int8)

    if {"systolic_bp", "diastolic_bp"}.issubset(df.columns):
        df["bp_hypertensive"] = ((df["systolic_bp"] >= 140) | (df["diastolic_bp"] >= 90)).astype(np.int8)

# --- Log transforms for heavy tails (works well with trees too) ---
    log_cols = [
        "triglycerides", "cholesterol_total", "ldl_cholesterol", "hdl_cholesterol",
        "tg_hdl_ratio", "ldl_hdl_ratio", "non_hdl"
    ]
    for c in log_cols:
        if c in df.columns:
            # ensure non-negative for log1p
            df[f"log1p_{c}"] = np.log1p(np.clip(df[c].astype(float), a_min=0, a_max=None))

    return df

train_fe = enhance_features(train)
test_fe = enhance_features(test)

new_cols = [c for c in train_fe.columns if c not in train.columns]
print("Added features:", new_cols)



X = train_fe.drop(columns=[TARGET, ID_COL])
y = train[TARGET].astype(int)

X_test_raw = test_fe.drop(columns=[ID_COL])
test_ids = test[ID_COL].copy()

categorical_features = [
"gender",
"ethnicity",
"education_level",
"income_level",
"smoking_status",
"employment_status",
]

#Ensure category dtype (good for LGBM, XGB categorical support, and stable memory)
for col in categorical_features:
    X[col] = X[col].astype("category")
    X_test_raw[col] = X_test_raw[col].astype("category")

numeric_features = [c for c in X.columns if c not in categorical_features]
print("Num features:", len(numeric_features))
print("Cat features:", len(categorical_features))



def fit_quantile_clipper(df: pd.DataFrame, cols, q_low=0.001, q_high=0.999):
    bounds = {}
    for c in cols:
        s = df[c].astype(float)
        lo = s.quantile(q_low)
        hi = s.quantile(q_high)
    # guard (can happen if constant / weird)
        if not np.isfinite(lo): lo = s.min()
        if not np.isfinite(hi): hi = s.max()
        if lo > hi: lo, hi = hi, lo
        bounds[c] = (lo, hi)
    return bounds

def apply_clipper(df: pd.DataFrame, bounds: dict):
    df = df.copy()
    for c, (lo, hi) in bounds.items():
        df[c] = df[c].astype(float).clip(lo, hi)
    return df


HARD_CLAMPS = {
"age": (0, 120),
"sleep_hours_per_day": (0, 24),
"screen_time_hours_per_day": (0, 24),
"waist_to_hip_ratio": (0.5, 1.5),
"bmi": (10, 70),
}

def apply_hard_clamps(df: pd.DataFrame):
    df = df.copy()
    for c, (lo, hi) in HARD_CLAMPS.items():
        if c in df.columns:
            df[c] = df[c].astype(float).clip(lo, hi)
    return df


kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

#Storage for base models
oof_lgb = np.zeros(len(X), dtype=float)
oof_xgb = np.zeros(len(X), dtype=float)
oof_cb = np.zeros(len(X), dtype=float)

test_lgb = np.zeros(len(X_test_raw), dtype=float)
test_xgb = np.zeros(len(X_test_raw), dtype=float)
test_cb = np.zeros(len(X_test_raw), dtype=float)

fold_scores = {"lgb": [], "xgb": [], "cb": []}


lgb_params = dict(
objective="binary",
metric="auc",
n_estimators=5000,
learning_rate=0.02,
num_leaves=64,
max_depth=-1,
min_child_samples=60,
subsample=0.8,
subsample_freq=1,
colsample_bytree=0.8,
reg_alpha=1.0,
reg_lambda=4.0,
min_split_gain=0.0,
n_jobs=-1,
random_state=SEED
)

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n===== Fold {fold}/{N_SPLITS} | LightGBM =====")
    X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]


# hard clamps first (domain safety)
    X_tr = apply_hard_clamps(X_tr)
    X_va = apply_hard_clamps(X_va)
    X_te = apply_hard_clamps(X_test_raw)

# quantile clipper fit on train fold only
    bounds = fit_quantile_clipper(X_tr[numeric_features], numeric_features, q_low=0.001, q_high=0.999)
    X_tr[numeric_features] = apply_clipper(X_tr[numeric_features], bounds)
    X_va[numeric_features] = apply_clipper(X_va[numeric_features], bounds)
    X_te_clipped = X_te.copy()
    X_te_clipped[numeric_features] = apply_clipper(X_te_clipped[numeric_features], bounds)

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(200),
            lgb.log_evaluation(200),
        ],
        categorical_feature=categorical_features
    )

    pred_va = model.predict_proba(X_va)[:, 1]
    pred_te = model.predict_proba(X_te_clipped)[:, 1]

    oof_lgb[va_idx] = pred_va
    test_lgb += pred_te / N_SPLITS

    auc = roc_auc_score(y_va, pred_va)
    fold_scores["lgb"].append(auc)
    print(f"Fold {fold} AUC: {auc:.6f} | best_iter={getattr(model, 'best_iteration_', None)}")

    del X_tr, X_va, X_te, X_te_clipped, model
    gc.collect()
    print("\nLightGBM CV AUC:", roc_auc_score(y, oof_lgb))
    print("Fold AUCs:", [round(s, 6) for s in fold_scores["lgb"]])


xgb_params = dict(
objective="binary:logistic",
eval_metric="auc",
n_estimators=6000,
learning_rate=0.02,
max_depth=5,
min_child_weight=8,
subsample=0.85,
colsample_bytree=0.85,
reg_alpha=2.0,
reg_lambda=6.0,
gamma=0.0,
tree_method="hist",
enable_categorical=True,
random_state=SEED,
)

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n===== Fold {fold}/{N_SPLITS} | XGBoost =====")
    X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]


# hard clamps + fold-safe quantile clipping
    X_tr = apply_hard_clamps(X_tr)
    X_va = apply_hard_clamps(X_va)
    X_te = apply_hard_clamps(X_test_raw)

    bounds = fit_quantile_clipper(X_tr[numeric_features], numeric_features, q_low=0.001, q_high=0.999)
    X_tr[numeric_features] = apply_clipper(X_tr[numeric_features], bounds)
    X_va[numeric_features] = apply_clipper(X_va[numeric_features], bounds)
    X_te_clipped = X_te.copy()
    X_te_clipped[numeric_features] = apply_clipper(X_te_clipped[numeric_features], bounds)

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=300,
        early_stopping_rounds=200,
    )

    pred_va = model.predict_proba(X_va)[:, 1]
    pred_te = model.predict_proba(X_te_clipped)[:, 1]

    oof_xgb[va_idx] = pred_va
    test_xgb += pred_te / N_SPLITS

    auc = roc_auc_score(y_va, pred_va)
    fold_scores["xgb"].append(auc)
    print(f"Fold {fold} AUC: {auc:.6f} | best_ntree_limit={getattr(model, 'best_iteration', None)}")

    del X_tr, X_va, X_te, X_te_clipped, model
    gc.collect()
    print("\nXGBoost CV AUC:", roc_auc_score(y, oof_xgb))
    print("Fold AUCs:", [round(s, 6) for s in fold_scores["xgb"]])


cb_params = dict(
loss_function="Logloss",
eval_metric="AUC",
iterations=6000,
learning_rate=0.02,
depth=8,
l2_leaf_reg=6.0,
random_seed=SEED,
verbose=300,
allow_writing_files=False
)

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n===== Fold {fold}/{N_SPLITS} | CatBoost =====")
    X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]


    X_tr = apply_hard_clamps(X_tr)
    X_va = apply_hard_clamps(X_va)
    X_te = apply_hard_clamps(X_test_raw)

    bounds = fit_quantile_clipper(X_tr[numeric_features], numeric_features, q_low=0.001, q_high=0.999)
    X_tr[numeric_features] = apply_clipper(X_tr[numeric_features], bounds)
    X_va[numeric_features] = apply_clipper(X_va[numeric_features], bounds)
    X_te_clipped = X_te.copy()
    X_te_clipped[numeric_features] = apply_clipper(X_te_clipped[numeric_features], bounds)

    train_pool = Pool(X_tr, label=y_tr, cat_features=categorical_features)
    valid_pool = Pool(X_va, label=y_va, cat_features=categorical_features)
    test_pool  = Pool(X_te_clipped, cat_features=categorical_features)

    model = CatBoostClassifier(**cb_params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    pred_va = model.predict_proba(valid_pool)[:, 1]
    pred_te = model.predict_proba(test_pool)[:, 1]

    oof_cb[va_idx] = pred_va
    test_cb += pred_te / N_SPLITS
    
    auc = roc_auc_score(y_va, pred_va)
    fold_scores["cb"].append(auc)
    print(f"Fold {fold} AUC: {auc:.6f} | best_iter={model.get_best_iteration()}")
    
    del X_tr, X_va, X_te, X_te_clipped, train_pool, valid_pool, test_pool, model
    gc.collect()
    print("\nCatBoost CV AUC:", roc_auc_score(y, oof_cb))
    print("Fold AUCs:", [round(s, 6) for s in fold_scores["cb"]])


auc_lgb = roc_auc_score(y, oof_lgb)
auc_xgb = roc_auc_score(y, oof_xgb)
auc_cb = roc_auc_score(y, oof_cb)

print(f"LGB AUC: {auc_lgb:.6f}")
print(f"XGB AUC: {auc_xgb:.6f}")
print(f"CB AUC: {auc_cb:.6f}")

#Try a few hand-tuned combos
candidates = [
(0.45, 0.45, 0.10),
(0.50, 0.30, 0.20),
(0.35, 0.50, 0.15),
(0.55, 0.25, 0.20),
(0.54, 0.23, 0.22),
(0.53, 0.24, 0.23)
]

best = (-1, None)
for wl, wx, wc in candidates:
    oof_ens = wl*oof_lgb + wx*oof_xgb + wc*oof_cb
    auc = roc_auc_score(y, oof_ens)
    print(f"weights (lgb,xgb,cb)=({wl:.2f},{wx:.2f},{wc:.2f}) -> AUC={auc:.6f}")
    if auc > best[0]:
        best = (auc, (wl, wx, wc))

print("\nBest weighted ensemble:", best)


base_oof = np.vstack([oof_lgb, oof_xgb, oof_cb]).T
base_test = np.vstack([test_lgb, test_xgb, test_cb]).T

def stacking_cv(base_oof, y, base_test, n_splits=5, seed=42):
    kf2 = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    meta_oof = np.zeros(len(y), dtype=float)
    meta_test = np.zeros(base_test.shape[0], dtype=float)


    for fold, (tr_idx, va_idx) in enumerate(kf2.split(base_oof, y), 1):
        X_tr, X_va = base_oof[tr_idx], base_oof[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        meta = LogisticRegression(
            C=0.3,
            max_iter=3000,
            solver="lbfgs"
        )
        meta.fit(X_tr, y_tr)

        pred_va = meta.predict_proba(X_va)[:, 1]
        meta_oof[va_idx] = pred_va

        meta_test += meta.predict_proba(base_test)[:, 1] / n_splits

        auc = roc_auc_score(y_va, pred_va)
        print(f"Meta fold {fold} AUC: {auc:.6f}")

    return meta_oof, meta_test
    
stack_oof, stack_test = stacking_cv(base_oof, y, base_test, n_splits=N_SPLITS, seed=SEED)
stack_auc = roc_auc_score(y, stack_oof)
print("\nStacking CV AUC:", stack_auc)


best_weight_auc, (wl, wx, wc) = best
ens_test = wl*test_lgb + wx*test_xgb + wc*test_cb

print(f"Best weighted ensemble AUC: {best_weight_auc:.6f}")
print(f"Stacking AUC: {stack_auc:.6f}")

if stack_auc >= best_weight_auc:
    final_pred = stack_test
    method = "stacking"
else:
    final_pred = ens_test
    method = f"weighted_{wl:.2f}{wx:.2f}{wc:.2f}"

sub = pd.DataFrame({ID_COL: test_ids, TARGET: final_pred})
sub.to_csv("submission.csv", index=False)

print(f"Saved submission.csv using: {method}")
display(sub.head())


# --- 1. SET YOUR WEIGHTS HERE ---
wl = 0.80  # Weight for LightGBM
wx = 0.00  # Weight for XGBoost
wc = 0.20  # Weight for CatBoost

# --- 2. CALCULATE PREDICTIONS ---
# We explicitly use '*' to multiply the weight by the prediction array
final_pred = (wl * test_lgb) + (wx * test_xgb) + (wc * test_cb)

# --- 3. CREATE & SAVE SUBMISSION ---
sub = pd.DataFrame({ID_COL: test_ids, TARGET: final_pred})
sub.to_csv("submission_m.csv", index=False)

print(f"Saved submission.csv using manual weights: LGB={wl}, XGB={wx}, CB={wc}")
display(sub.head())

