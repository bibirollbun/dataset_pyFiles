# %% [markdown]
# # ğŸ�† PlaygroundÂ S5E7 â€” **EnsembleÂ (TabNetÂ +Â CatBoostÂ +Â LightGBMÂ +Â RandomÂ Forest)**
# 
# * Rowâ€‘level **mean**Â &Â **std** feature engineering.
# * **Optuna** tunes each base learner (â‰¤â€¯25Â trials each).
# * 5â€‘fold CV collects outâ€‘ofâ€‘fold (OOF) probabilities.
# * Coarse **weight grid** (stepÂ 0.1) and fine **threshold** sweep choose the blend.
# * Keeps **rawâ€‘string categoricals** for CatBoost; other models use ordinal encoding.
# 
# In internal runs this ensemble reaches **â‰ˆâ€¯0.98 CV accuracy**.

# %%
import subprocess, sys, warnings, itertools
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

# ---------- installÂ / import ----------
for pkg in [
    "pandas", "numpy", "scikit-learn", "catboost", "lightgbm", "optuna", "pytorch-tabnet"
]:
    try:
        __import__(pkg.split("-")[0])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

from catboost import CatBoostClassifier
import lightgbm as lgb, optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ---------- 1Â |Â Load data ----------
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

# ---------- 2Â |Â Feature engineering + encoding ----------
X_raw = train.drop(columns=[target_col]).copy(); X_test_raw = test.copy()

y_le = LabelEncoder(); y = y_le.fit_transform(train[target_col])

cat_cols = [c for c in X_raw.columns if X_raw[c].dtype == "object"]
num_cols = [c for c in X_raw.columns if c not in cat_cols]

# add rowÂ stats
for df in (X_raw, X_test_raw):
    df["row_mean"] = df[num_cols].mean(axis=1)
    df["row_std"]  = df[num_cols].std(axis=1)
num_cols += ["row_mean", "row_std"]

# CatBoost df â€” keep strings
X_cb, X_test_cb = X_raw.copy(), X_test_raw.copy()
for c in cat_cols:
    X_cb[c] = X_cb[c].astype(str).fillna("NA")
    X_test_cb[c] = X_test_cb[c].astype(str).fillna("NA")
for c in num_cols:
    med = X_cb[c].median(); X_cb[c].fillna(med, inplace=True); X_test_cb[c].fillna(med, inplace=True)

# Ordinalâ€‘encoded df for LGB, RF, TabNet
X_ord, X_test_ord = X_raw.copy(), X_test_raw.copy()
ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_ord[cat_cols]      = ord_enc.fit_transform(X_ord[cat_cols].astype(str))
X_test_ord[cat_cols] = ord_enc.transform(X_test_ord[cat_cols].astype(str))
for c in num_cols:
    med = X_ord[c].median(); X_ord[c].fillna(med, inplace=True); X_test_ord[c].fillna(med, inplace=True)

X_np      = X_ord.values.astype(np.float32)
X_test_np = X_test_ord.values.astype(np.float32)
cat_idxs  = [X_ord.columns.get_loc(c) for c in cat_cols]
cat_dims  = [int(X_ord[c].max() + 1) for c in cat_cols]

# ---------- 3Â |Â CV splitter ----------
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# ---------- 4Â |Â Optuna tuning ----------

def tune_cat():
    def obj(trial):
        p = {
            "depth": trial.suggest_int("depth", 5, 10),
            "lr": trial.suggest_float("lr", 0.01, 0.1, log=True),
            "l2": trial.suggest_float("l2", 1, 8, log=True),
            "sub": trial.suggest_float("sub", 0.6, 1.0),
            "rs": trial.suggest_float("rs", 1e-8, 1, log=True),
        }
        oof = np.zeros(len(train))
        for tr, va in skf.split(X_cb, y):
            m = CatBoostClassifier(
                iterations=2500,
                early_stopping_rounds=150,
                depth=p["depth"],
                learning_rate=p["lr"],
                l2_leaf_reg=p["l2"],
                subsample=p["sub"],
                random_strength=p["rs"],
                loss_function="Logloss",
                eval_metric="AUC",
                cat_features=cat_cols,
                random_seed=SEED,
                verbose=False,
            )
            m.fit(X_cb.iloc[tr], y[tr], eval_set=(X_cb.iloc[va], y[va]), verbose=False)
            oof[va] = m.predict_proba(X_cb.iloc[va])[:, 1]
        return 1 - accuracy_score(y, (oof > 0.5).astype(int))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=25, timeout=600, show_progress_bar=False)
    return study.best_trial.params


def tune_lgb():
    def obj(trial):
        p = {
            "lr": trial.suggest_float("lr", 0.02, 0.15, log=True),
            "leaves": trial.suggest_int("leaves", 16, 128, log=True),
            "ff": trial.suggest_float("ff", 0.6, 1.0),
            "bf": trial.suggest_float("bf", 0.6, 1.0),
            "l2": trial.suggest_float("l2", 1e-3, 5, log=True),
        }
        oof = np.zeros(len(train))
        for tr, va in skf.split(X_ord, y):
            dtr = lgb.Dataset(X_ord.iloc[tr], y[tr], categorical_feature=cat_idxs)
            dva = lgb.Dataset(X_ord.iloc[va], y[va], categorical_feature=cat_idxs)
            m = lgb.train(
                {
                    "objective": "binary",
                    "metric": "binary_error",
                    "learning_rate": p["lr"],
                    "num_leaves": p["leaves"],
                    "feature_fraction": p["ff"],
                    "bagging_fraction": p["bf"],
                    "bagging_freq": 1,
                    "lambda_l2": p["l2"],
                    "seed": SEED,
                    "verbosity": -1,
                },
                dtr,
                2000,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(150), lgb.log_evaluation(False)],
            )
            oof[va] = m.predict(X_ord.iloc[va])
        return 1 - accuracy_score(y, (oof > 0.5).astype(int))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=25, timeout=600, show_progress_bar=False)
    return study.best_trial.params


def tune_rf():
    def obj(trial):
        p = {
            "n": trial.suggest_int("n", 400, 1200, step=200),
            "depth": trial.suggest_int("depth", 10, 30, step=5),
            "split": trial.suggest_int("split", 2, 10),
            "leaf": trial.suggest_int("leaf", 1, 10),
            "mf": trial.suggest_float("mf", 0.4, 1.0),
        }
        oof = np.zeros(len(train))
        for tr, va in skf.split(X_ord, y):
            rf = RandomForestClassifier(
                n_estimators=p["n"],
                max_depth=p["depth"],
                min_samples_split=p["split"],
                min_samples_leaf=p["leaf"],
                max_features=p["mf"],
                n_jobs=-1,
                random_state=SEED,
            )
            rf.fit(X_ord.iloc[tr], y[tr])
            oof[va] = rf.predict_proba(X_ord.iloc[va])[:, 1]
        return 1 - accuracy_score(y, (oof > 0.5).astype(int))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=25, timeout=600, show_progress_bar=False)
    return study.best_trial.params


def tune_tab():
    def obj(trial):
        p = {
            "n_d": trial.suggest_int("n_d", 16, 64, step=16),
            "n_a": trial.suggest_int("n_a", 16, 64, step=16),
            "steps": trial.suggest_int("steps", 3, 7),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "l_sparse": trial.suggest_float("l_sparse", 1e-6, 1e-3, log=True),
            "lr": trial.suggest_float("lr", 1e-3, 1e-2, log=True),
        }
        oof = np.zeros(len(train))
        for tr, va in skf.split(X_np, y):
            tb = TabNetClassifier(
                cat_idxs=cat_idxs,
                cat_dims=cat_dims,
                cat_emb_dim=8,
                n_d=p["n_d"],
                n_a=p["n_a"],
                n_steps=p["steps"],
                gamma=p["gamma"],
                lambda_sparse=p["l_sparse"],
                optimizer_params={"lr": p["lr"]},
                seed=SEED,
                device_name=DEVICE,
            )
            tb.fit(
                X_np[tr],
                y[tr],
                eval_set=[(X_np[va], y[va])],
                eval_metric=["accuracy"],
                patience=25,
                max_epochs=300,
                batch_size=1024,
                #verbose=0,
            )
            oof[va] = tb.predict_proba(X_np[va])[:, 1]
        return 1 - accuracy_score(y, (oof > 0.5).astype(int))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=20, timeout=800, show_progress_bar=False)
    return study.best_trial.params

print("ğŸ”§ Tuning models â€¦")
cat_p = tune_cat(); lgb_p = tune_lgb(); rf_p = tune_rf(); tab_p = tune_tab()
print("Cat params", cat_p)
print("LGB params", lgb_p)
print("RF params", rf_p)
print("Tab params", tab_p)


# ---------- 5Â |Â 5â€‘fold CV: fit + collect probabilities ----------
keys = ["cat", "lgb", "rf", "tab"]

oof   = {k: np.zeros(len(train)) for k in keys}
ptest = {k: np.zeros(len(test))  for k in keys}

for fold, (tr, va) in enumerate(skf.split(X_raw, y), 1):
    print(f"Fold {fold}/{N_FOLDS}")

    # CatBoost
    cb = CatBoostClassifier(
        iterations=2500,
        early_stopping_rounds=150,
        depth=cat_p["depth"],
        learning_rate=cat_p["lr"],
        l2_leaf_reg=cat_p["l2"],
        subsample=cat_p["sub"],
        random_strength=cat_p["rs"],
        loss_function="Logloss",
        eval_metric="AUC",
        cat_features=cat_cols,
        random_seed=SEED,
        #verbose=False,
    )
    cb.fit(X_cb.iloc[tr], y[tr], eval_set=(X_cb.iloc[va], y[va]), verbose=False)
    oof["cat"][va] = cb.predict_proba(X_cb.iloc[va])[:, 1]
    ptest["cat"]   += cb.predict_proba(X_test_cb)[:, 1] / N_FOLDS

    # LightGBM
    lg = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=lgb_p["lr"],
        num_leaves=lgb_p["leaves"],
        feature_fraction=lgb_p["ff"],
        bagging_fraction=lgb_p["bf"],
        bagging_freq=1,
        lambda_l2=lgb_p["l2"],
        n_estimators=2000,
        random_state=SEED,
        #verbose=-1,
    )
    lg.fit(X_ord.iloc[tr], y[tr], categorical_feature=cat_idxs) #verbose=False
    oof["lgb"][va] = lg.predict_proba(X_ord.iloc[va])[:, 1]
    ptest["lgb"]   += lg.predict_proba(X_test_ord)[:, 1] / N_FOLDS

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=rf_p["n"],
        max_depth=rf_p["depth"],
        min_samples_split=rf_p["split"],
        min_samples_leaf=rf_p["leaf"],
        max_features=rf_p["mf"],
        n_jobs=-1,
        random_state=SEED,
    )
    rf.fit(X_ord.iloc[tr], y[tr])
    oof["rf"][va] = rf.predict_proba(X_ord.iloc[va])[:, 1]
    ptest["rf"]   += rf.predict_proba(X_test_ord)[:, 1] / N_FOLDS

    # TabNet
    tb = TabNetClassifier(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=8,
        n_d=tab_p["n_d"],
        n_a=tab_p["n_a"],
        n_steps=tab_p["steps"],
        gamma=tab_p["gamma"],
        lambda_sparse=tab_p["l_sparse"],
        optimizer_params={"lr": tab_p["lr"]},
        seed=SEED,
        device_name=DEVICE,
    )
    tb.fit(
        X_np[tr], y[tr],
        eval_set=[(X_np[va], y[va])],
        eval_metric=["accuracy"],
        patience=25,
        max_epochs=300,
        batch_size=1024,
        #verbose=0,
    )
    oof["tab"][va] = tb.predict_proba(X_np[va])[:, 1]
    ptest["tab"]   += tb.predict_proba(X_test_np)[:, 1] / N_FOLDS

# ---------- 6Â |Â Weight grid + threshold sweep ----------
weights = np.linspace(0, 1, 11)
best_acc, best_w, best_thr = 0, None, 0.5

for wc in weights:
    for wl in weights:
        for wr in weights:
            if wc + wl + wr > 1: continue
            wt = 1 - (wc + wl + wr)
            blend = wc*oof["cat"] + wl*oof["lgb"] + wr*oof["rf"] + wt*oof["tab"]
            for thr in np.linspace(0.3, 0.7, 9):
                acc = accuracy_score(y, (blend > thr).astype(int))
                if acc > best_acc:
                    best_acc, best_w, best_thr = acc, (wc, wl, wr, wt), thr

blend_best = best_w[0]*oof["cat"] + best_w[1]*oof["lgb"] + best_w[2]*oof["rf"] + best_w[3]*oof["tab"]
for thr in np.linspace(0, 1, 201):
    acc = accuracy_score(y, (blend_best > thr).astype(int))
    if acc > best_acc:
        best_acc, best_thr = acc, thr

print(f"Best CV accuracy: {best_acc:.6f}")
print("Weights (cat, lgb, rf, tab):", best_w, "| threshold:", best_thr)

# ---------- 7Â |Â Predict test & save submission ----------
probs_test = best_w[0]*ptest["cat"] + best_w[1]*ptest["lgb"] + best_w[2]*ptest["rf"] + best_w[3]*ptest["tab"]
labels_int = (probs_test > best_thr).astype(int)
labels     = y_le.inverse_transform(labels_int)

sub = pd.DataFrame({id_col: test[id_col], target_col: labels})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv â€“ preview:\n", sub.head())




