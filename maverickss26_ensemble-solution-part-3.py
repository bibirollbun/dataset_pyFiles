# %% [markdown]
# # ğŸ�†Â PlaygroundÂ S5E7 â€” Ensembleâ€¯v8
# **CatBoostÂ +Â LightGBMÂ +Â XGBoostÂ +Â RandomÂ ForestÂ +Â ExtraÂ TreesÂ +Â MLP**
# 
# * Six diverse base learners.
# * 5â€‘fold CV generates outâ€‘ofâ€‘fold (OOF) probabilities.
# * Exhaustive **weight grid** (stepÂ 0.2) + fine **threshold sweep** maximise CV accuracy.
# * Saves `submission.csv` with the best blend.

# %%
import subprocess, sys, warnings, itertools, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

for pkg in ["pandas", "numpy", "scikit-learn", "catboost", "lightgbm", "xgboost"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

from catboost import CatBoostClassifier
import lightgbm as lgb, xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

SEED = 42
np.random.seed(SEED)

# --------------------------------------------------
# 1Â |Â Load data & identify columns
# --------------------------------------------------
DATA_DIR = "/kaggle/input/playground-series-s5e7"
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
extra = [c for c in train.columns if c not in test.columns]

id_col = next((c for c in extra if "id" in c.lower()), "id")
try:
    target_col = next(c for c in extra if train[c].nunique()==2 and c!=id_col)
except StopIteration:
    target_col = (set(extra)-{id_col}).pop()
print("ID:", id_col, "Target:", target_col)

# --------------------------------------------------
# 2Â |Â Preâ€‘processing
# --------------------------------------------------
X_raw = train.drop(columns=[target_col]).copy(); X_test_raw = test.copy()

y_le = LabelEncoder(); y = y_le.fit_transform(train[target_col])

cat_cols = [c for c in X_raw.columns if X_raw[c].dtype=="object"]
num_cols = [c for c in X_raw.columns if c not in cat_cols]

# CatBoost data (string cats)
X_cb, X_test_cb = X_raw.copy(), X_test_raw.copy()
for c in cat_cols:
    X_cb[c] = X_cb[c].astype(str).fillna("NA"); X_test_cb[c] = X_test_cb[c].astype(str).fillna("NA")
for c in num_cols:
    med = X_cb[c].median(); X_cb[c].fillna(med, inplace=True); X_test_cb[c].fillna(med, inplace=True)

# Ordinalâ€‘encoded data for tree models + MLP
X_enc, X_test_enc = X_raw.copy(), X_test_raw.copy()
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_enc[cat_cols] = enc.fit_transform(X_enc[cat_cols].astype(str))
X_test_enc[cat_cols] = enc.transform(X_test_enc[cat_cols].astype(str))
for c in num_cols:
    med = X_enc[c].median(); X_enc[c].fillna(med, inplace=True); X_test_enc[c].fillna(med, inplace=True)

cat_idx_lgb = [X_enc.columns.get_loc(c) for c in cat_cols]

# --------------------------------------------------
# 3Â |Â Fixed hyperâ€‘params (fast)
# --------------------------------------------------
cb_params = dict(iterations=2000, depth=7, learning_rate=0.05, l2_leaf_reg=3,
                 loss_function="Logloss", eval_metric="AUC", random_seed=SEED, verbose=False, cat_features=cat_cols)

lgb_params = dict(objective="binary", metric="binary_error", learning_rate=0.05, num_leaves=64,
                  feature_fraction=0.8, bagging_fraction=0.9, bagging_freq=1, random_state=SEED, verbosity=-1)

xgb_params = dict(objective="binary:logistic", eval_metric="error", learning_rate=0.05, max_depth=6,
                  subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0, tree_method="hist", random_state=SEED)

rf_params = dict(n_estimators=800, max_depth=20, max_features=0.7, n_jobs=-1, random_state=SEED)

et_params = dict(n_estimators=800, max_depth=20, max_features=0.7, n_jobs=-1, random_state=SEED)

mlp_params = dict(hidden_layer_sizes=(128,64), activation="relu", solver="adam", alpha=1e-3,
                  learning_rate_init=0.001, max_iter=300, random_state=SEED)

# --------------------------------------------------
# 4Â |Â 5â€‘fold CV â€“ gather OOF & test probs
# --------------------------------------------------
keys = ["cat","lgb","xgb","rf","et","mlp"]
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof   = {k: np.zeros(len(train)) for k in keys}
ptest = {k: np.zeros(len(test))  for k in keys}

for fold,(tr,va) in enumerate(skf.split(X_enc,y),1):
    print(f"Fold {fold}/{N_FOLDS}")

    cb = CatBoostClassifier(**cb_params); cb.fit(X_cb.iloc[tr],y[tr])
    oof["cat"][va] = cb.predict_proba(X_cb.iloc[va])[:,1]; ptest["cat"] += cb.predict_proba(X_test_cb)[:,1]/N_FOLDS

    lgbm = lgb.LGBMClassifier(**lgb_params, n_estimators=2000)
    lgbm.fit(X_enc.iloc[tr], y[tr], categorical_feature=cat_idx_lgb)
    oof["lgb"][va] = lgbm.predict_proba(X_enc.iloc[va])[:,1]; ptest["lgb"] += lgbm.predict_proba(X_test_enc)[:,1]/N_FOLDS

    xgbm = xgb.XGBClassifier(**xgb_params, n_estimators=2000)
    xgbm.fit(X_enc.iloc[tr], y[tr], verbose=False)
    oof["xgb"][va] = xgbm.predict_proba(X_enc.iloc[va])[:,1]; ptest["xgb"] += xgbm.predict_proba(X_test_enc)[:,1]/N_FOLDS

    rf = RandomForestClassifier(**rf_params); rf.fit(X_enc.iloc[tr], y[tr])
    oof["rf"][va]  = rf.predict_proba(X_enc.iloc[va])[:,1];  ptest["rf"]  += rf.predict_proba(X_test_enc)[:,1]/N_FOLDS

    et = ExtraTreesClassifier(**et_params); et.fit(X_enc.iloc[tr], y[tr])
    oof["et"][va]  = et.predict_proba(X_enc.iloc[va])[:,1];  ptest["et"]  += et.predict_proba(X_test_enc)[:,1]/N_FOLDS

    mlp = MLPClassifier(**mlp_params); mlp.fit(X_enc.iloc[tr], y[tr])
    oof["mlp"][va] = mlp.predict_proba(X_enc.iloc[va])[:,1]; ptest["mlp"] += mlp.predict_proba(X_test_enc)[:,1]/N_FOLDS

# --------------------------------------------------
# 5Â |Â Weight grid (stepÂ 0.2) + fine threshold
# --------------------------------------------------
weights = np.linspace(0,1,6)  # 0,0.2,â€¦1
best_acc, best_w, best_thr = 0, None, 0.5

for wc, wl, wx, wr, we in itertools.product(weights, repeat=5):
    if wc+wl+wx+wr+we > 1: continue
    wm = 1 - (wc+wl+wx+wr+we)
    blend = wc*oof["cat"] + wl*oof["lgb"] + wx*oof["xgb"] + wr*oof["rf"] + we*oof["et"] + wm*oof["mlp"]
    for thr in np.linspace(0.3,0.7,9):
        acc = accuracy_score(y, (blend>thr).astype(int))
        if acc > best_acc:
            best_acc, best_w, best_thr = acc, (wc,wl,wx,wr,we,wm), thr

# Fine threshold sweep for best weights
blend_best = sum(w*oof[k] for w,k in zip(best_w,keys))
for thr in np.linspace(0,1,201):
    acc = accuracy_score(y, (blend_best>thr).astype(int))
    if acc > best_acc:
        best_acc, best_thr = acc, thr

print("Best CV accuracy:", best_acc)
print("Weights (cat,lgb,xgb,rf,et,mlp):", best_w, "| threshold:", best_thr)

# --------------------------------------------------
# 6Â |Â Predict test & create submission
# --------------------------------------------------
probs_test = sum(w*ptest[k] for w,k in zip(best_w,keys))
labels_int = (probs_test>best_thr).astype(int)
labels     = y_le.inverse_transform(labels_int)

sub = pd.DataFrame({id_col: test[id_col], target_col: labels})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv â€“ preview:\n", sub.head())





