import os
import gc
import math
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
import xgboost as xgb
import catboost as cb


SEED = 42
N_FOLDS = 5
np.random.seed(SEED)


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sub   = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

TARGET = "y"
ID_COL = "id"


train_orig = train.copy()
test_orig  = test.copy()


cat_cols = ["job","marital","education","default","housing","loan","contact","month","poutcome"]
num_cols = [c for c in train.columns if c not in cat_cols + [ID_COL, TARGET]]


for c in num_cols:
    if train[c].isna().any() or test[c].isna().any():
        med = train[c].median()
        train[c] = train[c].fillna(med)
        test[c]  = test[c].fillna(med)


for c in cat_cols:
    if train[c].isna().any() or test[c].isna().any():
        mode_val = train[c].mode(dropna=True)[0]
        train[c] = train[c].fillna(mode_val)
        test[c]  = test[c].fillna(mode_val)


def safe_div(a, b):
    out = a / np.where(b == 0, np.nan, b)
    return np.nan_to_num(out, posinf=0.0, neginf=0.0)


month_map = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12
}


if train["month"].dtype == object:
    train["month_ord"] = train["month"].str.lower().map(month_map).fillna(0).astype(int)
    test["month_ord"]  = test["month"].str.lower().map(month_map).fillna(0).astype(int)
else:
    train["month_ord"] = train["month"]
    test["month_ord"]  = test["month"]


train["pdays_unknown"] = (train["pdays"] >= 999).astype(int)
test["pdays_unknown"]  = (test["pdays"]  >= 999).astype(int)

train["pdays_capped"] = train["pdays"].clip(upper=998)
test["pdays_capped"]  = test ["pdays"].clip(upper=998)


for col in ["balance","duration","campaign","previous","pdays_capped","age","day"]:
    train[f"{col}_log1p"] = np.log1p(np.maximum(train[col], 0))
    test [f"{col}_log1p"] = np.log1p(np.maximum(test[col],  0))


train["dur_per_call"] = safe_div(train["duration"], np.maximum(train["campaign"], 1))
test ["dur_per_call"] = safe_div(test ["duration"], np.maximum(test ["campaign"], 1))

train["prev_contacted"] = (train["previous"] > 0).astype(int)
test ["prev_contacted"] = (test ["previous"] > 0).astype(int)

train["bal_per_age"] = safe_div(train["balance"], np.maximum(train["age"], 1))
test ["bal_per_age"] = safe_div(test ["balance"], np.maximum(test ["age"], 1))


for c in cat_cols:
    freq = pd.concat([train[c], test[c]]).value_counts(dropna=False, normalize=True)
    train[f"{c}_freq"] = train[c].map(freq)
    test [f"{c}_freq"] = test [c].map(freq)


te_cats = ["job","marital","education","contact","poutcome","month"]  # tweakable


def cv_target_encode(train_df, test_df, cat_cols, target, n_folds=5, seed=SEED, smoothing=10.0):
    train_new = train_df.copy()
    test_new  = test_df.copy()

    global_mean = train_new[target].mean()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for c in cat_cols:
        oof_te = np.zeros(len(train_new))
        test_te_folds = []

        for tr_idx, va_idx in skf.split(train_new, train_new[target]):
            tr, va = train_new.iloc[tr_idx], train_new.iloc[va_idx]

            stats = tr.groupby(c)[target].agg(["mean","count"])
            smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
            oof_te[va_idx] = va[c].map(smooth).fillna(global_mean).values

            test_map = test_new[c].map(smooth).fillna(global_mean).values
            test_te_folds.append(test_map)

        train_new[f"TE_{c}"] = oof_te
        test_new[f"TE_{c}"] = np.mean(test_te_folds, axis=0)

    return train_new, test_new


train, test = cv_target_encode(train, test, te_cats, TARGET, n_folds=N_FOLDS, smoothing=20.0)


encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[c], test[c]], axis=0).astype(str))
    train[c] = le.transform(train[c].astype(str))
    test[c]  = le.transform(test[c].astype(str))
    encoders[c] = le


drop_cols = [TARGET, ID_COL]
features = [c for c in train.columns if c not in drop_cols]


X = train[features].copy()
y = train[TARGET].astype(int).values
X_test = test[features].copy()


X = X.replace([np.inf, -np.inf], 0).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cb  = np.zeros(len(X))


pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))
pred_cb  = np.zeros(len(X_test))


from lightgbm import early_stopping


from lightgbm import early_stopping
from sklearn.metrics import roc_auc_score

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    print("="*50)
    print(f"▶▶ Starting Fold {fold} ...")
    
    # Split data
    print(" Splitting train/validation data...")
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # ---- LightGBM ----
    print(" Training LightGBM model...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.01,
        num_leaves=64,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=SEED
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[early_stopping(stopping_rounds=100)]
    )
    print("  Finished LightGBM training")
    oof_lgb[va_idx] = lgb_model.predict_proba(X_va)[:,1]
    pred_lgb += lgb_model.predict_proba(X_test)[:,1] / N_FOLDS
    print("  Predictions saved for LightGBM")

    # ---- XGBoost ----
    print(" Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_estimators=5000,
        learning_rate=0.02,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=2.0,
        random_state=SEED
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        early_stopping_rounds=400,
        verbose=False
    )
    print("  Finished XGBoost training")
    oof_xgb[va_idx] = xgb_model.predict_proba(X_va)[:,1]
    pred_xgb += xgb_model.predict_proba(X_test)[:,1] / N_FOLDS
    print("  Predictions saved for XGBoost")

    # ---- CatBoost ----
    print(" Training CatBoost model...")
    cb_model = cb.CatBoostClassifier(
        iterations=5000,
        learning_rate=0.02,
        depth=7,
        eval_metric="AUC",
        random_seed=SEED,
        l2_leaf_reg=3.0,
        verbose=False,
        loss_function="Logloss",
        early_stopping_rounds=400
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, verbose=False)
    print("  Finished CatBoost training")
    oof_cb[va_idx] = cb_model.predict_proba(X_va)[:,1]
    pred_cb += cb_model.predict_proba(X_test)[:,1] / N_FOLDS
    print("  Predictions saved for CatBoost")

    # ---- Metrics ----
    auc_lgb = roc_auc_score(y_va, oof_lgb[va_idx])
    auc_xgb = roc_auc_score(y_va, oof_xgb[va_idx])
    auc_cb  = roc_auc_score(y_va,  oof_cb[va_idx])
    print(f" ✅ Fold {fold} completed")
    print(f"    AUCs -> LGB: {auc_lgb:.6f} | XGB: {auc_xgb:.6f} | CB: {auc_cb:.6f}")
    print("="*50)


auc_l = roc_auc_score(y, oof_lgb)
auc_x = roc_auc_score(y, oof_xgb)
auc_c = roc_auc_score(y, oof_cb)
print(f"\nOOF AUCs  -> LGB: {auc_l:.6f} | XGB: {auc_x:.6f} | CB: {auc_c:.6f}")


oof_stack_in = np.vstack([oof_lgb, oof_xgb, oof_cb]).T
test_stack_in = np.vstack([pred_lgb, pred_xgb, pred_cb]).T


best_w = np.array([1/3, 1/3, 1/3]) 


try:
    import optuna
    def objective(trial):
        w1 = trial.suggest_float("w1", 0.0, 1.0)
        w2 = trial.suggest_float("w2", 0.0, 1.0)
        w3 = trial.suggest_float("w3", 0.0, 1.0)
        s = w1 + w2 + w3 + 1e-12
        w = np.array([w1/s, w2/s, w3/s])
        blend = (oof_stack_in * w).sum(axis=1)
        return 1.0 - roc_auc_score(y, blend)  # minimize (1 - AUC)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=80, show_progress_bar=False)
    w = study.best_params
    s = w["w1"] + w["w2"] + w["w3"] + 1e-12
    best_w = np.array([w["w1"]/s, w["w2"]/s, w["w3"]/s])
    print(f"Optuna best weights: {best_w}")
except Exception as e:
    print(f"Optuna not available or failed ({e}). Using equal weights.")


blend_oof = (oof_stack_in * best_w).sum(axis=1)
blend_test = (test_stack_in * best_w).sum(axis=1)
print(f"Weighted Blend OOF AUC: {roc_auc_score(y, blend_oof):.6f}")


stacker = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    n_jobs=-1 if hasattr(LogisticRegression, "n_jobs") else None
)


stacker.fit(oof_stack_in, y)
oof_meta = stacker.predict_proba(oof_stack_in)[:,1]
test_meta = stacker.predict_proba(test_stack_in)[:,1]
print(f"Stacker (LR) OOF AUC: {roc_auc_score(y, oof_meta):.6f}")


final_oof  = 0.5 * blend_oof + 0.5 * oof_meta
final_pred = 0.5 * blend_test + 0.5 * test_meta
print(f"\nFINAL OOF AUC: {roc_auc_score(y, final_oof):.6f}")


sub["y"] = final_pred
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv")


del train, test, X, X_test
gc.collect()




