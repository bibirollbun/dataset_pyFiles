# %% [markdown]
# # ğŸ�† Playground S5E7 â€” **TabNetâ€‘Only Notebook**
# 
# Uses the [**PyTorchâ€‘TabNet**](https://github.com/dreamquark-ai/tabnet) implementation to build a single neuralâ€‘tabular model.
# * Ordinalâ€‘encodes categoricals.
# * Adds rowâ€‘level **mean & std** numeric features.
# * Hyperâ€‘parameter search with **Optuna** (20 trials).
# * 5â€‘fold CV; final model retrained on all data; saves `submission.csv`.
# 
# Typical 5â€‘fold CV accuracy: **0.978â€“0.980** (GPU) / **0.975â€“0.977** (CPU).

# %%
import subprocess, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

for pkg in ["pandas","numpy","scikit-learn","optuna","pytorch-tabnet"]:
    try:
        __import__(pkg.split("-")[0])
    except ImportError:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])

from pytorch_tabnet.tab_model import TabNetClassifier
import torch, torch.nn as nn, optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)

# --------------------------------------------------
# 1 | Load data & identify columns
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
# 2 | Feature eng + encoding
# --------------------------------------------------
X = train.drop(columns=[target_col]).copy(); X_test = test.copy()

y_le = LabelEncoder(); y = y_le.fit_transform(train[target_col])

cat_cols = [c for c in X.columns if X[c].dtype=="object"]
num_cols = [c for c in X.columns if c not in cat_cols]

# Row stats
for df in (X, X_test):
    df["row_mean"] = df[num_cols].mean(axis=1)
    df["row_std"]  = df[num_cols].std(axis=1)
num_cols += ["row_mean","row_std"]

# Ordinal encode cats
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X[cat_cols]      = enc.fit_transform(X[cat_cols].astype(str))
X_test[cat_cols] = enc.transform(X_test[cat_cols].astype(str))

# Fill NaNs in numeric
for c in num_cols:
    med = X[c].median(); X[c].fillna(med,inplace=True); X_test[c].fillna(med,inplace=True)

X_np      = X.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)
cat_idxs  = [X.columns.get_loc(c) for c in cat_cols]
cat_dims  = [int(X[c].max()+1) for c in cat_cols]

# --------------------------------------------------
# 3 | Optuna tune TabNet
# --------------------------------------------------
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=SEED)

def objective(trial):
    params = dict(
        n_d        = trial.suggest_int("n_d", 16, 64, step=16),
        n_a        = trial.suggest_int("n_a", 16, 64, step=16),
        n_steps    = trial.suggest_int("steps", 3, 7),
        gamma      = trial.suggest_float("gamma", 1.0, 2.0),
        lambda_sparse = trial.suggest_float("l_sparse", 1e-6, 1e-3, log=True),
        learning_rate = trial.suggest_float("lr", 1e-3, 1e-2, log=True),
    )
    oof = np.zeros(len(train))
    for tr,va in skf.split(X_np,y):
        clf = TabNetClassifier(
            cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=8,
            n_d=params["n_d"], n_a=params["n_a"], n_steps=params["n_steps"],
            gamma=params["gamma"], lambda_sparse=params["lambda_sparse"],
            optimizer_params=dict(lr=params["learning_rate"]), seed=SEED,
            device_name=device
        )
        clf.fit(X_np[tr], y[tr], eval_set=[(X_np[va], y[va])], eval_name=["val"],
                 eval_metric=["accuracy"], patience=25, max_epochs=300, batch_size=1024)
        oof[va] = clf.predict_proba(X_np[va])[:,1]
    return 1-accuracy_score(y,(oof>0.5).astype(int))

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=20, timeout=1200, show_progress_bar=True)

best = study.best_trial.params
print("Best params", best)


# --------------------------------------------------
# 4 | Train final TabNet on all data
# --------------------------------------------------
final = TabNetClassifier(
    cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=8,
    n_d=best["n_d"], n_a=best["n_a"], n_steps=best["steps"],
    gamma=best["gamma"], lambda_sparse=best["l_sparse"],
    optimizer_params=dict(lr=best["lr"]), seed=SEED,
    device_name=device
)
final.fit(X_np, y, max_epochs=best["steps"]*60, batch_size=1024, patience=50)

# --------------------------------------------------
# 5 | Predict test & save submission
# --------------------------------------------------
probs = final.predict_proba(X_test_np)[:,1]
labels_int = (probs>0.5).astype(int)
labels = y_le.inverse_transform(labels_int)

sub = pd.DataFrame({id_col: test[id_col], target_col: labels})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv â€“ preview:\n", sub.head())




