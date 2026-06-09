# %% [markdown]
# # ğŸ�† Playground S5E7 â€” Ensemble v9 (Complete)
# **CatBoost + LightGBM + XGBoost + Random Forest**
# 
# * Rowâ€‘wise *mean* and *std* feature engineering.
# * **Optuna** tunes each base learner (â‰¤â€¯25 trials each).
# * 5â€‘fold CV gathers OOF predictions.
# * Exhaustive weight grid (0.1 step) + fine threshold sweep chooses the best blend.
# * Target CV â‰¥ **0.98**.

# %%
import subprocess, sys, warnings, itertools, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

for pkg in ["pandas","numpy","scikit-learn","catboost","lightgbm","xgboost","optuna"]:
    try: __import__(pkg)
    except ImportError: subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])

from catboost import CatBoostClassifier
import lightgbm as lgb, xgboost as xgb, optuna
from sklearn.ensemble import RandomForestClassifier
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
# 2Â |Â Feature engineering
# --------------------------------------------------
X_raw = train.drop(columns=[target_col]).copy(); X_test_raw = test.copy()

y_le = LabelEncoder(); y = y_le.fit_transform(train[target_col])

cat_cols = [c for c in X_raw.columns if X_raw[c].dtype=="object"]
num_cols = [c for c in X_raw.columns if c not in cat_cols]

# Row statistics
for df in (X_raw, X_test_raw):
    df["row_mean"] = df[num_cols].mean(axis=1)
    df["row_std"]  = df[num_cols].std(axis=1)
num_cols += ["row_mean","row_std"]

# CatBoost data (string cats)
X_cb, X_test_cb = X_raw.copy(), X_test_raw.copy()
for c in cat_cols:
    X_cb[c] = X_cb[c].astype(str).fillna("NA"); X_test_cb[c] = X_test_cb[c].astype(str).fillna("NA")
for c in num_cols:
    med = X_cb[c].median(); X_cb[c].fillna(med, inplace=True); X_test_cb[c].fillna(med, inplace=True)

# Ordinalâ€‘encoded data
X_enc, X_test_enc = X_raw.copy(), X_test_raw.copy()
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_enc[cat_cols]      = enc.fit_transform(X_enc[cat_cols].astype(str))
X_test_enc[cat_cols] = enc.transform(X_test_enc[cat_cols].astype(str))
for c in num_cols:
    med = X_enc[c].median(); X_enc[c].fillna(med, inplace=True); X_test_enc[c].fillna(med, inplace=True)

cat_idx_lgb = [X_enc.columns.get_loc(c) for c in cat_cols]

# --------------------------------------------------
# 3Â |Â Optuna tuning helpers (each â‰¤Â 25 trials)
# --------------------------------------------------
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def tune_cat():
    def obj(trial):
        params = dict(
            depth=trial.suggest_int("depth",5,10),
            learning_rate=trial.suggest_float("lr",0.01,0.1,log=True),
            l2_leaf_reg=trial.suggest_float("l2",1,8,log=True),
            subsample=trial.suggest_float("sub",0.6,1.0),
            random_strength=trial.suggest_float("rs",1e-8,1,log=True),
            loss_function="Logloss", eval_metric="AUC", random_seed=SEED,
            verbose=False, cat_features=cat_cols,
        )
        oof=np.zeros(len(train))
        for tr,va in skf.split(X_cb,y):
            m=CatBoostClassifier(**params, iterations=2500, early_stopping_rounds=150)
            m.fit(X_cb.iloc[tr],y[tr],eval_set=(X_cb.iloc[va],y[va]), verbose=False)
            oof[va]=m.predict_proba(X_cb.iloc[va])[:,1]
        return 1-accuracy_score(y,(oof>0.5).astype(int))
    study=optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj,n_trials=25,timeout=600,show_progress_bar=False)
    p=study.best_trial.params
    return dict(iterations=2500,early_stopping_rounds=150,depth=p["depth"],learning_rate=p["lr"],
                l2_leaf_reg=p["l2"],subsample=p["sub"],random_strength=p["rs"],
                loss_function="Logloss",eval_metric="AUC",cat_features=cat_cols,random_seed=SEED,verbose=False)


def tune_lgb():
    def obj(trial):
        params=dict(objective="binary",metric="binary_error",seed=SEED,verbosity=-1,
                    learning_rate=trial.suggest_float("lr",0.02,0.15,log=True),
                    num_leaves=trial.suggest_int("leaves",16,128,log=True),
                    feature_fraction=trial.suggest_float("ff",0.6,1.0),
                    bagging_fraction=trial.suggest_float("bf",0.6,1.0),bagging_freq=1,
                    lambda_l2=trial.suggest_float("l2",1e-3,5,log=True))
        oof=np.zeros(len(train))
        for tr,va in skf.split(X_enc,y):
            dtr=lgb.Dataset(X_enc.iloc[tr],y[tr],categorical_feature=cat_idx_lgb)
            dva=lgb.Dataset(X_enc.iloc[va],y[va],categorical_feature=cat_idx_lgb)
            m=lgb.train(params,dtr,2000,valid_sets=[dva],callbacks=[lgb.early_stopping(150),lgb.log_evaluation(False)])
            oof[va]=m.predict(X_enc.iloc[va])
        return 1-accuracy_score(y,(oof>0.5).astype(int))
    study=optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj,n_trials=25,timeout=600,show_progress_bar=False)
    p=study.best_trial.params
    return dict(objective="binary",metric="binary_error",seed=SEED,verbosity=-1,learning_rate=p["lr"],
                num_leaves=p["leaves"],feature_fraction=p["ff"],bagging_fraction=p["bf"],bagging_freq=1,lambda_l2=p["l2"])


def tune_xgb():
    def obj(trial):
        params=dict(objective="binary:logistic",eval_metric="error",tree_method="hist",seed=SEED,
                    learning_rate=trial.suggest_float("lr",0.02,0.15,log=True),
                    max_depth=trial.suggest_int("depth",4,10),
                    subsample=trial.suggest_float("sub",0.6,1.0),
                    colsample_bytree=trial.suggest_float("col",0.6,1.0),
                    reg_lambda=trial.suggest_float("l2",1e-3,5,log=True),
                    reg_alpha=trial.suggest_float("l1",1e-3,5,log=True))
        oof=np.zeros(len(train))
        for tr,va in skf.split(X_enc,y):
            clf=xgb.XGBClassifier(**params,n_estimators=2000)
            clf.fit(X_enc.iloc[tr],y[tr],eval_set=[(X_enc.iloc[va],y[va])],early_stopping_rounds=150,verbose=False)
            oof[va]=clf.predict_proba(X_enc.iloc[va])[:,1]
        return 1-accuracy_score(y,(oof>0.5).astype(int))
    study=optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj,n_trials=25,timeout=600,show_progress_bar=False)
    p=study.best_trial.params
    return dict(objective="binary:logistic",eval_metric="error",tree_method="hist",seed=SEED,
                learning_rate=p["lr"],max_depth=p["depth"],subsample=p["sub"],
                colsample_bytree=p["col"],reg_lambda=p["l2"],reg_alpha=p["l1"])


def tune_rf():
    def obj(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n", 400, 1200, step=200),
            max_depth         = trial.suggest_int("depth", 10, 30, step=5),
            min_samples_split = trial.suggest_int("split", 2, 10),
            min_samples_leaf  = trial.suggest_int("leaf", 1, 10),
            max_features      = trial.suggest_float("mf", 0.4, 1.0),
            n_jobs            = -1,
            random_state      = SEED,
        )
        oof = np.zeros(len(train))
        for tr_idx, va_idx in skf.split(X_enc, y):
            rf = RandomForestClassifier(**params)
            rf.fit(X_enc.iloc[tr_idx], y[tr_idx])
            oof[va_idx] = rf.predict_proba(X_enc.iloc[va_idx])[:, 1]
        return 1 - accuracy_score(y, (oof > 0.5).astype(int))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=25, timeout=600, show_progress_bar=False)
    bp = study.best_trial.params
    return dict(n_estimators=bp["n"], max_depth=bp["depth"], min_samples_split=bp["split"],
                min_samples_leaf=bp["leaf"], max_features=bp["mf"], n_jobs=-1, random_state=SEED)

# --------------------------------------------------
# 4 | Tune models (this can take ~10â€‘15 min CPU)
# --------------------------------------------------
print("Tuning CatBoost â€¦");  cat_params = tune_cat()
print("Tuning LightGBM â€¦"); lgb_params = tune_lgb()
print("Tuning XGBoost  â€¦"); xgb_params = tune_xgb()
print("Tuning RandomForest â€¦"); rf_params = tune_rf()

# --------------------------------------------------
# 5 | 5â€‘fold CV with tuned hyperâ€‘params â€” gather OOF & test probs
# --------------------------------------------------
keys = ["cat", "lgb", "xgb", "rf"]
oof   = {k: np.zeros(len(train)) for k in keys}
ptest = {k: np.zeros(len(test))  for k in keys}

for fold, (tr, va) in enumerate(skf.split(X_enc, y), 1):
    print(f"Fold {fold}/{N_FOLDS}")

    cb = CatBoostClassifier(**cat_params)
    cb.fit(X_cb.iloc[tr], y[tr], eval_set=(X_cb.iloc[va], y[va]), verbose=False)
    oof["cat"][va] = cb.predict_proba(X_cb.iloc[va])[:, 1]
    ptest["cat"]   += cb.predict_proba(X_test_cb)[:, 1] / N_FOLDS

    lg = lgb.LGBMClassifier(**lgb_params, n_estimators=2000)
    lg.fit(X_enc.iloc[tr], y[tr], categorical_feature=cat_idx_lgb)
    oof["lgb"][va] = lg.predict_proba(X_enc.iloc[va])[:, 1]
    ptest["lgb"]   += lg.predict_proba(X_test_enc)[:, 1] / N_FOLDS

    xg = xgb.XGBClassifier(**xgb_params, n_estimators=2000)
    xg.fit(X_enc.iloc[tr], y[tr], eval_set=[(X_enc.iloc[va], y[va])], verbose=False)
    oof["xgb"][va] = xg.predict_proba(X_enc.iloc[va])[:, 1]
    ptest["xgb"]   += xg.predict_proba(X_test_enc)[:, 1] / N_FOLDS

    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_enc.iloc[tr], y[tr])
    oof["rf"][va]  = rf.predict_proba(X_enc.iloc[va])[:, 1]
    ptest["rf"]    += rf.predict_proba(X_test_enc)[:, 1] / N_FOLDS

# --------------------------------------------------
# 6 | Weight grid search (0.1 step) + fine threshold
# --------------------------------------------------
weights = np.linspace(0, 1, 11)
best_acc, best_w, best_thr = 0, None, 0.5

for wc, wl in itertools.product(weights, repeat=2):
    for wx in weights:
        if wc + wl + wx > 1: continue
        wr = 1 - (wc + wl + wx)
        blend = wc * oof["cat"] + wl * oof["lgb"] + wx * oof["xgb"] + wr * oof["rf"]
        for thr in np.linspace(0.3, 0.7, 9):
            acc = accuracy_score(y, (blend > thr).astype(int))
            if acc > best_acc:
                best_acc, best_w, best_thr = acc, (wc, wl, wx, wr), thr

# fine threshold sweep
blend_best = sum(w * oof[k] for w, k in zip(best_w, keys))
for thr in np.linspace(0, 1, 201):
    acc = accuracy_score(y, (blend_best > thr).astype(int))
    if acc > best_acc:
        best_acc, best_thr = acc, thr

print("Best CV accuracy:", best_acc)
print("Weights (cat, lgb, xgb, rf):", best_w, "| threshold:", best_thr)

# --------------------------------------------------
# 7 | Predict test & save submission
# --------------------------------------------------
probs_test = sum(w * ptest[k] for w, k in zip(best_w, keys))
labels_int = (probs_test > best_thr).astype(int)
labels     = y_le.inverse_transform(labels_int)

sub = pd.DataFrame({id_col: test[id_col], target_col: labels})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv â€“ preview:", sub.head())



sub = pd.read_csv('submission.csv')


sub.head()




