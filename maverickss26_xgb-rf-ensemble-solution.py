# %% [markdown]
# # ğŸ�† PlaygroundÂ S5E7 â€” **Twoâ€‘Model Ensemble (XGBoostÂ +Â RandomÂ Forest)**
# 
# * Ordinalâ€‘encodes all categorical features.
# * Adds rowâ€‘level **mean** & **std** numeric features.
# * **Optuna** performs comprehensive hyperâ€‘parameter tuning:
#   * **XGBoost:** 50Â trials, wide search over depth, learningÂ rate, subsample, colsample, reg_Î»/Î±.
#   * **RandomÂ Forest:** 50Â trials, grid over trees, depth, leaf, split, maxâ€‘features.
# * 5â€‘fold CV collects outâ€‘ofâ€‘fold (OOF) probabilities.
# * Optimises a single blending weight (0â€“1, stepÂ 0.02) and a decision threshold (0â€“1, stepÂ 0.01) to maximise CV accuracy.
# * Saves `submission.csv`.

# %%
import subprocess, sys, warnings, itertools, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

for pkg in ["pandas","numpy","scikit-learn","xgboost","optuna"]:
    try: __import__(pkg.split("-")[0])
    except ImportError: subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])

import xgboost as xgb, optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

SEED = 42
np.random.seed(SEED)

# --------------------------------------------------
# 1Â |Â Load data & basic columns
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
# 2Â |Â Feature eng + ordinal encoding
# --------------------------------------------------
X = train.drop(columns=[target_col]).copy(); X_test = test.copy()

y_le = LabelEncoder(); y = y_le.fit_transform(train[target_col])

cat_cols = [c for c in X.columns if X[c].dtype=="object"]
num_cols = [c for c in X.columns if c not in cat_cols]

# row stats
#for df in (X, X_test):
#    df["row_mean"] = df[num_cols].mean(axis=1)
#    df["row_std"]  = df[num_cols].std(axis=1)
#num_cols += ["row_mean","row_std"]

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X[cat_cols]      = enc.fit_transform(X[cat_cols].astype(str))
X_test[cat_cols] = enc.transform(X_test[cat_cols].astype(str))

for c in num_cols:
    med = X[c].median(); X[c].fillna(med,inplace=True); X_test[c].fillna(med,inplace=True)

# ---- frequency (log-count) encoding -----------
for col in cat_cols:
    freq = X[col].value_counts()
    X[f"{col}_freq"]      = X[col].map(freq).fillna(1).apply(np.log1p)
    X_test[f"{col}_freq"] = X_test[col].map(freq).fillna(1).apply(np.log1p)

# --------------------------------------------------
# 3Â |Â CV splitter
# --------------------------------------------------
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# --------------------------------------------------
# 4Â |Â Optuna tuning functions
# --------------------------------------------------

def tune_xgb():
    def obj(trial):
        params = dict(
            objective="binary:logistic",
            eval_metric="error",
            tree_method="hist",
            learning_rate=trial.suggest_float("lr", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("depth", 3, 10),
            subsample=trial.suggest_float("sub", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("col", 0.5, 1.0),
            reg_lambda=trial.suggest_float("l2", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("l1", 1e-3, 10.0, log=True),
            seed=SEED,
        )
        oof = np.zeros(len(train))
        for tr, va in skf.split(X, y):
            clf = xgb.XGBClassifier(**params, n_estimators=3000)
            clf.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], early_stopping_rounds=200, verbose=False)
            oof[va] = clf.predict_proba(X.iloc[va])[:, 1]
        return 1 - accuracy_score(y, (oof > 0.5).astype(int))

    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=50, timeout=1200, show_progress_bar=True)
    return st.best_trial.params


def tune_rf():
    def obj(trial):
        params = dict(
            n_estimators=trial.suggest_int("n", 500, 2000, step=250),
            max_depth=trial.suggest_int("depth", 10, 40, step=5),
            min_samples_split=trial.suggest_int("split", 2, 10),
            min_samples_leaf=trial.suggest_int("leaf", 1, 10),
            max_features=trial.suggest_float("mf", 0.3, 1.0),
            n_jobs=-1,
            random_state=SEED,
        )
        oof = np.zeros(len(train))
        for tr, va in skf.split(X, y):
            rf = RandomForestClassifier(**params)
            rf.fit(X.iloc[tr], y[tr]); oof[va] = rf.predict_proba(X.iloc[va])[:, 1]
        return 1 - accuracy_score(y, (oof > 0.5).astype(int))

    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=50, timeout=1200, show_progress_bar=True)
    return st.best_trial.params

print("ğŸ”§ Tuning XGBoost â€¦"); xgb_p = tune_xgb()
print("ğŸ”§ Tuning RandomForest â€¦"); rf_p = tune_rf()
print("Best XGB", xgb_p)
print("Best RF", rf_p)

# --------------------------------------------------
# 5Â |Â 5â€‘fold CV gather OOF / test probs
# --------------------------------------------------
keys = ["xgb", "rf"]
oof   = {k: np.zeros(len(train)) for k in keys}
ptest = {k: np.zeros(len(test))  for k in keys}

for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}/{N_FOLDS}")

    # XGBoost
    xgb_clf = xgb.XGBClassifier(**xgb_p, n_estimators=3000)
    xgb_clf.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], early_stopping_rounds=200, verbose=False)
    oof["xgb"][va] = xgb_clf.predict_proba(X.iloc[va])[:, 1]
    ptest["xgb"]   += xgb_clf.predict_proba(X_test)[:, 1] / N_FOLDS

    # Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=rf_p["n"],
        max_depth=rf_p["depth"],
        min_samples_split=rf_p["split"],
        min_samples_leaf=rf_p["leaf"],
        max_features=rf_p["mf"],
        n_jobs=-1,
        random_state=SEED,
    )
    rf_clf.fit(X.iloc[tr], y[tr])
    oof["rf"][va] = rf_clf.predict_proba(X.iloc[va])[:, 1]
    ptest["rf"]   += rf_clf.predict_proba(X_test)[:, 1] / N_FOLDS

# --------------------------------------------------
# 6Â |Â Blend weight + threshold search
# --------------------------------------------------
#best_acc, best_w, best_thr = 0, 0.5, 0.5
#for w in np.linspace(0, 1, 51):  # stepÂ 0.02
#    blend = w * oof["xgb"] + (1 - w) * oof["rf"]
#    for thr in np.linspace(0, 1, 101):
#        acc = accuracy_score(y, (blend > thr).astype(int))
#        if acc > best_acc:
#            best_acc, best_w, best_thr = acc, w, thr
#print(f"Best CV accuracy: {best_acc:.6f} | XGB weight={best_w:.2f} | threshold={best_thr:.2f}")

# --------------------------------------------------
# 7Â |Â Predict test & save
# --------------------------------------------------
#probs_test = best_w * ptest["xgb"] + (1 - best_w) * ptest["rf"]
#labels_int = (probs_test > best_thr).astype(int)
#labels     = y_le.inverse_transform(labels_int)

#sub = pd.DataFrame({id_col: test[id_col], target_col: labels})
#sub.to_csv("submission.csv", index=False)
#print("Saved submission.csv â€“ preview:\n", sub.head())


# --------------------------------------------------
# 6 |   âœ¨ Metaâ€‘stack instead of weight blend
# --------------------------------------------------
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# assemble metaâ€‘training set
X_meta = np.column_stack([oof["xgb"], oof["rf"]])

meta = Pipeline([
    ("sc", StandardScaler()),          # helpful: probabilities are not N(0,1)
    ("lr", LogisticRegressionCV(
        Cs=np.logspace(-3, 3, 15),     # wide grid
        cv=5,
        scoring="accuracy",
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
    )),
])
meta.fit(X_meta, y)

oof_stack = meta.predict_proba(X_meta)[:, 1]
print("StackÂ CV accuracy:", accuracy_score(y, (oof_stack > 0.5).astype(int)))

# meta predictions for TEST
X_test_meta = np.column_stack([ptest["xgb"], ptest["rf"]])
ptest_stack = meta.predict_proba(X_test_meta)[:, 1]

# --------------------------------------------------
# 7a |   Fineâ€‘search optimal threshold for the stack
# --------------------------------------------------
best_acc, best_thr = 0, 0.5
for thr in np.linspace(0, 1, 201):      # 0.005 steps
    acc = accuracy_score(y, (oof_stack > thr).astype(int))
    if acc > best_acc:
        best_acc, best_thr = acc, thr

print(f"Best CV accuracy: {best_acc:.6f} | threshold={best_thr:.3f}")

# --------------------------------------------------
# 7b |   Predict test & save
# --------------------------------------------------
labels_int = (ptest_stack > best_thr).astype(int)
labels     = y_le.inverse_transform(labels_int)

sub = pd.DataFrame({id_col: test[id_col], target_col: labels})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv â€“ preview:\\n", sub.head())

