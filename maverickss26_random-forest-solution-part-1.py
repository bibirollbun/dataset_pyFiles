# %% [markdown]
# # ğŸ�† Playgroundâ€¯S5E7Â â€” **Randomâ€¯Forestâ€‘Only Solution** (clean)
# 
# * Fast, torchâ€‘free baseline.
# * Ordinalâ€‘encodes categoricals.
# * Optuna tunes 5 key RF hyperâ€‘params with 5â€‘fold CV.
# * Retrains on all data â†’ saves `submission.csv`.

# %%
import subprocess, sys, warnings
warnings.filterwarnings("ignore")

for pkg in ["pandas", "numpy", "scikit-learn", "optuna"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import numpy as np, pandas as pd, optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

SEED = 42
np.random.seed(SEED)

# ------------------------------------------------------------------
# 1Â |Â Load data & detect ID / target
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 2Â |Â Ordinalâ€‘encode features
# ------------------------------------------------------------------
X      = train.drop(columns=[target_col]).copy()
X_test = test.copy()

y_le = LabelEncoder(); y = y_le.fit_transform(train[target_col])

cat_cols = [c for c in X.columns if X[c].dtype=="object"]
num_cols = [c for c in X.columns if c not in cat_cols]

for c in num_cols:
    med = X[c].median()
    X[c].fillna(med, inplace=True)
    X_test[c].fillna(med, inplace=True)

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X[cat_cols]      = enc.fit_transform(X[cat_cols].astype(str))
X_test[cat_cols] = enc.transform(X_test[cat_cols].astype(str))

# ------------------------------------------------------------------
# 3Â |Â Optuna hyperâ€‘param search
# ------------------------------------------------------------------
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def objective(trial):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 300, 1200, step=300),
        max_depth         = trial.suggest_int("max_depth", 5, 30, step=5),
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 10),
        max_features      = trial.suggest_float("max_features", 0.3, 1.0),
        random_state      = SEED,
        n_jobs            = -1,
    )

    oof = np.zeros(len(train))
    for tr, va in skf.split(X, y):
        rf = RandomForestClassifier(**params)
        rf.fit(X.iloc[tr], y[tr])
        oof[va] = rf.predict(X.iloc[va])

    acc = accuracy_score(y, oof)
    return 1.0 - acc

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=30, timeout=400, show_progress_bar=True)

best_params = study.best_trial.params
best_cv_acc  = 1.0 - study.best_value
print("Best CV accuracy:", best_cv_acc, "| params:", best_params)

# ------------------------------------------------------------------
# 4Â |Â Train final model & submit
# ------------------------------------------------------------------
rf_final = RandomForestClassifier(**best_params)
rf_final.fit(X, y)

pred_int = rf_final.predict(X_test)
labels   = y_le.inverse_transform(pred_int)

sub = pd.DataFrame({id_col: test[id_col], target_col: labels})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv  â€” preview:\n", sub.head())




