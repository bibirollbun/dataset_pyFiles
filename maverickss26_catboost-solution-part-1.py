# Single‑model CatBoost that matches leaderboard top scores:
# 1. **Stratified 5‑fold CV** over a *tiny* hyper‑param grid.
# 2. **Threshold optimisation** on out‑of‑fold probabilities.
# 3. Encodes the target via **LabelEncoder** so `accuracy_score` gets matching types, then maps predictions back to original labels for submission.

import subprocess, sys, warnings
warnings.filterwarnings("ignore")

for pkg in ["pandas", "numpy", "catboost", "scikit-learn"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

SEED = 42
np.random.seed(SEED)

# %% [markdown]
# ## 1  |  Load Data

# %%
DATA_DIR = "/kaggle/input/playground-series-s5e7"
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
print(train.shape, test.shape)

# %% [markdown]
# ## 2  |  Identify `id` & `target`

# %%
extra = [c for c in train.columns if c not in test.columns]

def find_cols(extra):
    id_col, tgt_col = None, None
    for c in extra:
        if "id" in c.lower():
            id_col = c
        elif train[c].nunique() == 2:
            tgt_col = c
    id_col = id_col or "id"
    tgt_col = tgt_col or (set(extra) - {id_col}).pop()
    return id_col, tgt_col

id_col, target_col = find_cols(extra)
print("ID:", id_col, "| Target:", target_col)

# %% [markdown]
# ## 3  |  Pre‑processing & Label Encoding

# %%
X = train.drop(columns=[target_col]).copy()
y_raw = train[target_col].copy()

# ------- Encode target to 0/1 -------
le = LabelEncoder()
y = le.fit_transform(y_raw)  # 0/1 ints
print("Encoded classes:", le.classes_)

# ------- Basic feature NA handling -------
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]

for c in cat_cols:
    X[c]    = X[c].astype(str).fillna("NA")
    test[c] = test[c].astype(str).fillna("NA")
for c in num_cols:
    med = X[c].median()
    X[c]    = X[c].fillna(med)
    test[c] = test[c].fillna(med)

# %% [markdown]
# ## 4  |  Tiny Grid + 5‑Fold CV

# %%
param_grid = [
    dict(depth=6, learning_rate=0.07, l2_leaf_reg=2),
    dict(depth=7, learning_rate=0.05, l2_leaf_reg=3),
    dict(depth=8, learning_rate=0.04, l2_leaf_reg=5),
]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

best_score, best_params, best_thresh = 0, None, 0.5

for params in param_grid:
    print("\nTesting params:", params)
    oof = np.zeros(len(train))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        clf = CatBoostClassifier(
            loss_function="Logloss", eval_metric="Accuracy", iterations=2500,
            random_seed=SEED, verbose=False, cat_features=cat_cols, **params
        )
        clf.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict_proba(X.iloc[va_idx])[:, 1]
    for thr in np.linspace(0.4, 0.6, 41):
        acc = accuracy_score(y, (oof > thr).astype(int))
        if acc > best_score:
            best_score, best_params, best_thresh = acc, params, thr
    print(f"CV best so far: {best_score:.6f} @ thr={best_thresh:.3f}")

print("\nSelected params:", best_params, "| threshold:", best_thresh)

# %% [markdown]
# ## 5  |  Train Final Model

# %%
final_clf = CatBoostClassifier(
    loss_function="Logloss", eval_metric="Accuracy", iterations=2500,
    random_seed=SEED, verbose=200, cat_features=cat_cols, **best_params
)
final_clf.fit(X, y)

# %% [markdown]
# ## 6  |  Predict Test & Submission

# %%
probs = final_clf.predict_proba(test)[:, 1]
labels_int = (probs > best_thresh).astype(int)
labels = le.inverse_transform(labels_int)

sub = pd.DataFrame({id_col: test[id_col], target_col: labels})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv | Preview:\n", sub.head())


submission = pd.read_csv('submission.csv')
submission.head()




