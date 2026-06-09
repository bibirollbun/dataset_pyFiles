# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings, os, math, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
subbmission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


train


test


TARGET = "BeatsPerMinute"
feature_cols = [c for c in train.columns if c != TARGET]

print("Shapes -> train:", train.shape, " test:", test.shape)
train.head()


# Step 2 — basic checks
def show_missing(df, name):
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss>0]
    if len(miss)==0:
        print(f"[{name}] No missing values.")
    else:
        display(pd.DataFrame({"missing": miss, "pct": miss/len(df)*100}).head(20))

print("dtypes:")
print(train.dtypes)
print("\nMissing in train:"); show_missing(train, "train")
print("\nMissing in test:");  show_missing(test, "test")

print("\nDuplicate rows in train:", train.duplicated().sum())


# Step 3 — target distribution + stats
y = train[TARGET]
display(y.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]))

plt.figure()
plt.hist(y, bins=50)
plt.title("BeatsPerMinute distribution")
plt.xlabel("BPM"); plt.ylabel("Count")
plt.show()



# Step 4 — feature stats
X = train.drop(columns=[TARGET])
display(train[feature_cols].describe().T)

# correlations with target
corr = train[feature_cols].corrwith(train[TARGET]).sort_values(key=np.abs, ascending=False)
print("Correlation with BPM (abs sorted):")
display(corr)

plt.figure()
corr.abs().plot(kind="bar")
plt.ylabel("|corr|"); plt.title("Feature ↔ BPM correlation (abs)")
plt.tight_layout(); plt.show()

# scatter/hexbin (handles big data better than scatter)
for c in feature_cols:
    if c == "id": 
        continue
    plt.figure()
    plt.hexbin(train[c], train[TARGET], gridsize=40)
    plt.xlabel(c); plt.ylabel("BPM"); plt.title(f"{c} vs BPM")
    plt.show()



# === A) FEATURE ENGINEERING ===
import numpy as np, pandas as pd

DATA_DIR = "/kaggle/input/playground-series-s5e9"
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

TARGET = "BeatsPerMinute"
BASE   = [c for c in train.columns if c not in ("id", TARGET)]

def fe(df):
    D = df[BASE].copy()
    # robust clipping (0.5–99.5%)
    ql, qh = D.quantile(0.005), D.quantile(0.995)
    D = D.clip(lower=ql, upper=qh, axis=1)

    # log duration; a few gentle transforms
    if "TrackDurationMs" in D: D["TrackDurationMs_log"] = np.log1p(D["TrackDurationMs"])
    if "Energy" in D:         D["Energy_sqrt"] = np.sqrt(np.clip(D["Energy"], 0, None))
    if "MoodScore" in D:      D["MoodScore_sqrt"] = np.sqrt(np.clip(D["MoodScore"], 0, None))

    # simple interactions
    def has(*cols): return all(c in D.columns for c in cols)
    if has("Energy","MoodScore"):                  D["Energy_x_Mood"] = D["Energy"]*D["MoodScore"]
    if has("RhythmScore","Energy"):                D["Rhythm_x_Energy"] = D["RhythmScore"]*D["Energy"]
    if has("VocalContent","AcousticQuality"):      D["Vocal_x_Acoustic"] = D["VocalContent"]*D["AcousticQuality"]
    if has("InstrumentalScore","VocalContent"):    D["Instr_x_NotVocal"] = D["InstrumentalScore"]*(1 - D["VocalContent"])
    if has("LivePerformanceLikelihood","Energy"):  D["Live_x_Energy"] = D["LivePerformanceLikelihood"]*D["Energy"]
    if has("Energy","RhythmScore"):                D["Energy_div_Rhythm"] = D["Energy"]/(D["RhythmScore"]+1e-6)
    return D

X_full = fe(train).astype("float32")
Xt_full = fe(test).astype("float32")
y_full = train[TARGET].astype("float32").values
X_full.shape, Xt_full.shape



# === B1) Utilities ===
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

def rmse(y,p): 
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y,p,squared=False)

def make_regressor():
    return HistGradientBoostingRegressor(
        max_depth=6, learning_rate=0.08, max_iter=350, random_state=42
    )

# Train cluster-specific regressors on a given train split, then predict on its val split
def fit_predict_cluster_moe(Xtr, ytr, Xva, n_clusters=4, min_rows=500):
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zva = scaler.transform(Xtr), scaler.transform(Xva)

    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    kmeans.fit(Ztr)

    tr_labels = kmeans.predict(Ztr)
    va_labels = kmeans.predict(Zva)

    # global fallback model
    global_reg = make_regressor().fit(Xtr, ytr)

    regs = {}
    for c in range(n_clusters):
        idx = np.where(tr_labels == c)[0]
        if len(idx) < min_rows:
            regs[c] = global_reg  # too small → fallback
        else:
            regs[c] = make_regressor().fit(Xtr[idx], ytr[idx])

    preds = np.empty(len(Xva), dtype=np.float32)
    for c in range(n_clusters):
        idx = np.where(va_labels == c)[0]
        if len(idx):
            preds[idx] = regs[c].predict(Xva[idx])
    return preds, {"scaler": scaler, "kmeans": kmeans, "regs": regs}

# CV over different K to pick the best
def cv_cluster_moe(X, y, k_list=(3,4,5,6), n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    for k in k_list:
        fold_rmses = []
        for tr, va in kf.split(X):
            p, _ = fit_predict_cluster_moe(X[tr], y[tr], X[va], n_clusters=k)
            fold_rmses.append(rmse(y[va], p))
        results.append((k, float(np.mean(fold_rmses)), float(np.std(fold_rmses))))
        print(f"K={k} → RMSE {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}")
    results = pd.DataFrame(results, columns=["K","RMSE","SD"]).sort_values("RMSE")
    return results

cluster_results = cv_cluster_moe(X_full.values, y_full, k_list=(3,4,5,6), n_splits=3)
cluster_results



best_k = int(cluster_results.iloc[0]["K"])
print("Best K:", best_k)

# fit on full data
_ = StandardScaler().fit(X_full.values)
scaler = _
Z_full = scaler.transform(X_full.values)
Z_test = scaler.transform(Xt_full.values)

kmeans = KMeans(n_clusters=best_k, n_init="auto", random_state=42).fit(Z_full)
labels_full = kmeans.predict(Z_full)

# train per-cluster regs
regs = {}
global_reg = make_regressor().fit(X_full.values, y_full)
for c in range(best_k):
    idx = np.where(labels_full == c)[0]
    regs[c] = make_regressor().fit(X_full.values[idx], y_full[idx]) if len(idx)>=500 else global_reg

# predict test
test_labels = kmeans.predict(Z_test)
pred_test = np.empty(len(Xt_full), dtype=np.float32)
for c in range(best_k):
    idx = np.where(test_labels == c)[0]
    if len(idx): pred_test[idx] = regs[c].predict(Xt_full.values[idx])

sub_kmeans = pd.DataFrame({"id": test["id"], "BeatsPerMinute": pred_test})
sub_kmeans.to_csv("submission.csv", index=False)
sub_kmeans.head()


