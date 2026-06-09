import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings("ignore")



# path dataset di Kaggle
PATH = "/kaggle/input/playground-series-s5e12/"

train = pd.read_csv(PATH + "train.csv")
test  = pd.read_csv(PATH + "test.csv")
sample_sub = pd.read_csv(PATH + "sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape :", test.shape)

train.head()


TARGET = "diagnosed_diabetes"
ID_COL  = "id"

print(train[TARGET].value_counts(normalize=True))
print("\nMissing values (train):")
print(train.isna().sum().sort_values(ascending=False).head(20))

print("\nMissing values (test):")
print(test.isna().sum().sort_values(ascending=False).head(20))



# daftar fitur (semua kolom kecuali id dan target)
features = [c for c in train.columns if c not in [TARGET, ID_COL]]

X = train[features].copy()
X_test = test[features].copy()
y = train[TARGET].astype(int)

# deteksi kolom kategori dan numerik
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in features if c not in cat_cols]

print("Jumlah fitur total   :", len(features))
print("Fitur numerik        :", len(num_cols))
print("Fitur kategorikal    :", len(cat_cols))

# 1) Tangani missing value untuk numerik
if len(num_cols) > 0:
    for c in num_cols:
        median_val = X[c].median()
        X[c] = X[c].fillna(median_val)
        X_test[c] = X_test[c].fillna(median_val)

# 2) Tangani missing value untuk kategori (kalau ada)
if len(cat_cols) > 0:
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("missing")
        X_test[c] = X_test[c].astype(str).fillna("missing")
    
    # Ordinal encode kategori → angka
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[cat_cols] = oe.fit_transform(X[cat_cols])
    X_test[cat_cols] = oe.transform(X_test[cat_cols])
else:
    oe = None

print("Shape X     :", X.shape)
print("Shape X_test:", X_test.shape)



N_FOLDS = 10
RANDOM_STATE = 42

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# array untuk OOF prediction
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

# array untuk pred test
pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))
pred_cat = np.zeros(len(X_test))



for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n========== FOLD {fold} / {N_FOLDS} ==========")
    
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # -----------------------------
    # LightGBM
    # -----------------------------
    lgb_model = LGBMClassifier(
        n_estimators=4000,
        learning_rate=0.02,
        objective="binary",
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=-1,
        num_leaves=32,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[]
    )

    val_pred_lgb = lgb_model.predict_proba(X_val)[:, 1]
    test_pred_lgb = lgb_model.predict_proba(X_test)[:, 1]

    oof_lgb[val_idx] = val_pred_lgb
    pred_lgb += test_pred_lgb / N_FOLDS

    print("LGBM AUC:", roc_auc_score(y_val, val_pred_lgb))

    # -----------------------------
    # XGBoost
    # -----------------------------
    xgb_model = XGBClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    val_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
    test_pred_xgb = xgb_model.predict_proba(X_test)[:, 1]

    oof_xgb[val_idx] = val_pred_xgb
    pred_xgb += test_pred_xgb / N_FOLDS

    print("XGB  AUC:", roc_auc_score(y_val, val_pred_xgb))

    # -----------------------------
    # CatBoost
    # -----------------------------
    cat_model = CatBoostClassifier(
        depth=6,
        learning_rate=0.03,
        n_estimators=2500,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=False
    )

    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val)
    )

    val_pred_cat = cat_model.predict_proba(X_val)[:, 1]
    test_pred_cat = cat_model.predict_proba(X_test)[:, 1]

    oof_cat[val_idx] = val_pred_cat
    pred_cat += test_pred_cat / N_FOLDS

    print("CAT  AUC:", roc_auc_score(y_val, val_pred_cat))



print("=== OOF AUC per model ===")
auc_lgb = roc_auc_score(y, oof_lgb)
auc_xgb = roc_auc_score(y, oof_xgb)
auc_cat = roc_auc_score(y, oof_cat)

print(f"LGBM : {auc_lgb:.6f}")
print(f"XGB  : {auc_xgb:.6f}")
print(f"CAT  : {auc_cat:.6f}")

# weight bisa kamu tuning
w_lgb = 0.4
w_xgb = 0.3
w_cat = 0.3

oof_blend = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cat * oof_cat
auc_blend = roc_auc_score(y, oof_blend)

print("\n=== BLEND OOF AUC ===")
print(f"Blend (LGB {w_lgb}, XGB {w_xgb}, CAT {w_cat}) : {auc_blend:.6f}")



oof_df = pd.DataFrame({
    ID_COL: train[ID_COL],
    TARGET: y,
    "oof_lgb": oof_lgb,
    "oof_xgb": oof_xgb,
    "oof_cat": oof_cat,
    "oof_blend": oof_blend
})

oof_df.to_csv("oof_base_models.csv", index=False)
oof_df.head()



# pred test untuk blend sama seperti OOF
pred_blend = w_lgb * pred_lgb + w_xgb * pred_xgb + w_cat * pred_cat

sub_blend = sample_sub.copy()
sub_blend[TARGET] = pred_blend

sub_blend.to_csv("submission_blend_base.csv", index=False)
sub_blend.head()



X_meta = oof_df[["oof_lgb", "oof_xgb", "oof_cat"]]
y_meta = oof_df[TARGET]

meta_lr = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

meta_lr.fit(X_meta, y_meta)

oof_stack = meta_lr.predict_proba(X_meta)[:, 1]
auc_stack = roc_auc_score(y_meta, oof_stack)

print("AUC stacking (LR di atas 3 model):", auc_stack)



# susun fitur meta untuk test (pakai pred test masing2 model)
X_meta_test = pd.DataFrame({
    "oof_lgb": pred_lgb,
    "oof_xgb": pred_xgb,
    "oof_cat": pred_cat
})

pred_stack = meta_lr.predict_proba(X_meta_test)[:, 1]

sub_stack = sample_sub.copy()
sub_stack[TARGET] = pred_stack

sub_stack.to_csv("submission_stack_lr.csv", index=False)
sub_stack.head()

