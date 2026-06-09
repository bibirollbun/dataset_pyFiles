# === Robust CPU Ensemble for Playground S5E8 (Bank Binary Classification) ===
# Base models: LightGBM + CatBoost + OneHot Logistic Regression
# Meta model: LogisticRegressionCV stacking on OOF predictions
# Output: submission_stack.csv (id,y)

import os, gc, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# --- Load ---
DATA_DIR = "/kaggle/input"
cand = [d for d in os.listdir(DATA_DIR) if "bank" in d.lower() or "playground" in d.lower()]
COMP_DIR = os.path.join(DATA_DIR, cand[0]) if cand else DATA_DIR
train = pd.read_csv(os.path.join(COMP_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(COMP_DIR, "test.csv"))

# --- Identify columns ---
id_col = next((c for c in train.columns if c.lower()=="id" or c.lower().startswith("id")), train.columns[0])
target = next(c for c in train.columns if c in ("y","target","label"))
feat_cols = [c for c in train.columns if c not in (id_col, target)]

# --- Detect categoricals: object + low-card ints ---
cat_cols = []
for c in feat_cols:
    if train[c].dtype == "object":
        cat_cols.append(c)
    else:
        nunq = int(train[c].nunique(dropna=True))
        if str(train[c].dtype).startswith(("int","uint")) and 2 <= nunq <= 32:
            cat_cols.append(c)
num_cols = [c for c in feat_cols if c not in cat_cols]

# Cast to category for LGBM/CatBoost
for c in cat_cols:
    train[c] = train[c].astype("category")
    test[c]  = test[c].astype("category")

X, y = train[feat_cols], train[target].astype(int).values
X_test = test[feat_cols]

n_train, n_test = len(train), len(test)
oof_lgb = np.zeros(n_train, dtype=float); te_lgb = np.zeros(n_test, dtype=float)
oof_cb  = np.zeros(n_train, dtype=float); te_cb  = np.zeros(n_test, dtype=float)
oof_lr  = np.zeros(n_train, dtype=float); te_lr  = np.zeros(n_test, dtype=float)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- 1) LightGBM (callbacks API for LGBM>=4) ---
lgb_params = dict(
    objective="binary", boosting_type="gbdt",
    learning_rate=0.045, num_leaves=96, max_depth=-1,
    min_child_samples=80, subsample=0.85, subsample_freq=1,
    colsample_bytree=0.85, reg_alpha=0.0, reg_lambda=2.0,
    n_estimators=12000, n_jobs=-1, random_state=42
)

lgb_scores = []
for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    y_tr, y_va = y[tr], y[va]
    m = LGBMClassifier(**lgb_params)
    m.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(400), lgb.log_evaluation(200)]
    )
    pva = m.predict_proba(X_va)[:,1]; pte = m.predict_proba(X_test)[:,1]
    oof_lgb[va] = pva; te_lgb += pte / skf.n_splits
    auc = roc_auc_score(y_va, pva); lgb_scores.append(auc)
    print(f"[LGBM F{fold}] AUC={auc:.6f} | best_iter={m.best_iteration_}")
print(f"LGBM CV AUC: {np.mean(lgb_scores):.6f} ± {np.std(lgb_scores):.6f}")

# --- 2) CatBoost (CPU, early stopping) ---
# indices for cat features (CatBoost can take names or indices)
cat_idx = [X.columns.get_loc(c) for c in cat_cols] if cat_cols else []
cb_params = dict(
    loss_function="Logloss", eval_metric="AUC",
    depth=6, learning_rate=0.045, l2_leaf_reg=3.0,
    iterations=20000, random_seed=42, thread_count=-1,
    od_type="Iter", od_wait=600, verbose=200
)

cb_scores = []
for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    y_tr, y_va = y[tr], y[va]
    m = CatBoostClassifier(**cb_params)
    m.fit(X_tr, y_tr, eval_set=(X_va, y_va),
          cat_features=cat_idx, use_best_model=True)
    pva = m.predict_proba(X_va)[:,1]; pte = m.predict_proba(X_test)[:,1]
    oof_cb[va] = pva; te_cb += pte / skf.n_splits
    auc = roc_auc_score(y_va, pva); cb_scores.append(auc)
    print(f"[CatB F{fold}] AUC={auc:.6f}")
print(f"CatBoost CV AUC: {np.mean(cb_scores):.6f} ± {np.std(cb_scores):.6f}")

# --- 3) One-Hot Logistic Regression (adds a very different bias) ---
from scipy import sparse
pre = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True), cat_cols),
        ("num", "passthrough", num_cols)
    ],
    sparse_threshold=1.0
)
lr_scores = []
for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    y_tr, y_va = y[tr], y[va]
    pipe = make_pipeline(
        pre,
        LogisticRegression(
            solver="saga", C=2.0, max_iter=2000, n_jobs=-1, random_state=42
        )
    )
    pipe.fit(X_tr, y_tr)
    pva = pipe.predict_proba(X_va)[:,1]; pte = pipe.predict_proba(X_test)[:,1]
    oof_lr[va] = pva; te_lr += pte / skf.n_splits
    auc = roc_auc_score(y_va, pva); lr_scores.append(auc)
    print(f"[LogReg F{fold}] AUC={auc:.6f}")
print(f"LogReg CV AUC: {np.mean(lr_scores):.6f} ± {np.std(lr_scores):.6f}")

# --- Weighted average (by OOF AUC) as a strong baseline blend ---
aucs = np.array([np.mean(lgb_scores), np.mean(cb_scores), np.mean(lr_scores)])
w = aucs / aucs.sum()
oof_blend = w[0]*oof_lgb + w[1]*oof_cb + w[2]*oof_lr
te_blend  = w[0]*te_lgb  + w[1]*te_cb  + w[2]*te_lr
print("Weighted-blend OOF AUC:", roc_auc_score(y, oof_blend), "| weights:", w)

# --- Stacking meta-learner (often +0.001–0.003 AUC over simple blend) ---
stack_train = np.vstack([oof_lgb, oof_cb, oof_lr]).T
stack_test  = np.vstack([te_lgb,  te_cb,  te_lr ]).T
meta = LogisticRegressionCV(
    Cs=10, cv=5, scoring="roc_auc",
    max_iter=5000, n_jobs=-1, refit=True, random_state=42
)
meta.fit(stack_train, y)
oof_stack = meta.predict_proba(stack_train)[:,1]
te_stack  = meta.predict_proba(stack_test)[:,1]
print("Stack OOF AUC:", roc_auc_score(y, oof_stack))

# --- Save submission ---
sub = pd.DataFrame({"id": test[id_col], "y": te_stack})
sub.to_csv("submission_stack.csv", index=False)
print("Saved -> submission_stack.csv")

# (Optional) also save the simple weighted blend:
pd.DataFrame({"id": test[id_col], "y": te_blend}).to_csv("submission_blend.csv", index=False)
print("Saved -> submission_blend.csv")

del m; gc.collect()





