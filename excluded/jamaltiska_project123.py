
# ==== Setup ====
import os, gc, sys, math, random
import numpy as np
import pandas as pd
from pathlib import Path

# LightGBM (install if missing)
try:
    import lightgbm as lgb
except Exception:
    import pip
    pip.main(["-q", "install", "lightgbm"])
    import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

SEED = 42
random.seed(SEED); np.random.seed(SEED)

# Find ICR directory in /kaggle/input (attached competition folder)
def find_icr_dir():
    base = Path("/kaggle/input")
    for d in base.glob("*"):
        if (d/"train.csv").exists() and (d/"test.csv").exists() and (d/"sample_submission.csv").exists():
            return d
    raise FileNotFoundError("Could not locate ICR files under /kaggle/input. Attach the competition in 'Add Input'.")

DATA_DIR = find_icr_dir()
print("ICR data path:", DATA_DIR)
print("Files:", [p.name for p in DATA_DIR.glob("*.csv")])






# ==== Load ====
train = pd.read_csv(DATA_DIR/"train.csv")
test  = pd.read_csv(DATA_DIR/"test.csv")
sub   = pd.read_csv(DATA_DIR/"sample_submission.csv")
print("Shapes:", train.shape, test.shape)
display(train.head(3))




# ==== Identify target / id ====
TARGET_COL = [c for c in train.columns if c.lower() in ("class","target","label")]
assert len(TARGET_COL)==1, f"Couldn't identify target column; got {TARGET_COL}"
TARGET_COL = TARGET_COL[0]

ID_COL = [c for c in train.columns if c.lower() in ("id","id_code","row_id")]
ID_COL = ID_COL[0] if len(ID_COL)==1 else None
print("TARGET_COL:", TARGET_COL, "| ID_COL:", ID_COL)




# ==== Basic preprocessing ====
# Encode object/categorical columns jointly on train+test
cat_cols = [c for c in train.columns if train[c].dtype=="object"]
for c in cat_cols:
    le = LabelEncoder()
    vals = pd.concat([train[c].astype(str), test[c].astype(str)], axis=0)
    le.fit(vals.fillna("NA"))
    train[c] = le.transform(train[c].astype(str).fillna("NA"))
    test[c]  = le.transform(test[c].astype(str).fillna("NA"))

# Median-impute numerics
features_all = [c for c in train.columns if c not in ([TARGET_COL] + ([ID_COL] if ID_COL else []))]
for c in features_all:
    if pd.api.types.is_numeric_dtype(train[c]):
        med = train[c].median()
        train[c] = train[c].fillna(med)
        test[c]  = test[c].fillna(med)




# ==== Train with Stratified K-Fold LightGBM ====
FEATURES = [c for c in train.columns if c not in ([TARGET_COL] + ([ID_COL] if ID_COL else []))]
X = train[FEATURES].copy()
y = train[TARGET_COL].astype(int).copy()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof = np.zeros(len(train), dtype=float)
preds = np.zeros(len(test), dtype=float)

params = dict(
    objective="binary",
    boosting_type="gbdt",
    metric="binary_logloss",
    learning_rate=0.02,
    num_leaves=64,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_child_samples=40,
    n_estimators=5000,
    verbose=-1,
    random_state=SEED,
)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nFold {fold}")
    X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)]
    )

    oof[val_idx] = model.predict_proba(X_val)[:, 1]
    preds += model.predict_proba(test[FEATURES])[:, 1] / skf.n_splits

oof_clipped = np.clip(oof, 1e-7, 1-1e-7)
score = log_loss(y, oof_clipped)
print(f"\nOOF LogLoss: {score:.6f}")





# ==== Build submission (Id, class_0, class_1) ====
probs = np.clip(preds, 1e-7, 1-1e-7)

out = sub[['Id']].copy()
if 'Id' in test.columns and len(test)==len(sub):
    m = pd.Series(probs, index=test['Id']).to_dict()
    out['class_1'] = out['Id'].map(m).astype(float)
    if out['class_1'].isna().any():
        out['class_1'] = probs
else:
    out['class_1'] = probs

out['class_0'] = 1.0 - out['class_1']
out = out[['Id','class_0','class_1']]
assert out.notna().all().all(), "Submission contains NaNs"

out_path = "/kaggle/working/submission_icr.csv"
out.to_csv(out_path, index=False)
print("Saved:", out_path, "| shape:", out.shape)



# ==== FINAL VALIDATION + SAVE (ICR) ====
import os, numpy as np, pandas as pd
from pathlib import Path

# 1) Find the attached competition folder and load sample_submission for the expected shape
DATA_DIR = None
for d in Path("/kaggle/input").glob("*"):
    if (d/"sample_submission.csv").exists():
        DATA_DIR = d
        break
assert DATA_DIR is not None, "Could not find /kaggle/input/*/sample_submission.csv"
ssub = pd.read_csv(DATA_DIR/"sample_submission.csv")   # expected columns: Id, class_0, class_1

# 2) preds must be the probability for class_1 from your CV loop
p1 = np.asarray(preds, dtype="float64")
p1 = np.clip(p1, 1e-9, 1 - 1e-9)  # keep strictly in (0,1)

# 3) Align predictions to the exact Id order in sample_submission
if "Id" in test.columns and len(test) == len(ssub):
    order_map = pd.Series(p1, index=test["Id"])
    p1_aligned = ssub["Id"].map(order_map).to_numpy()
    # if any Ids didn't map (NaN), fall back to original order
    if np.isnan(p1_aligned).any():
        p1_aligned = p1
else:
    p1_aligned = p1

# 4) Build output with EXACT columns and types
out = pd.DataFrame({
    "Id": ssub["Id"],                   # keep Kaggle's Id dtype/order
    "class_0": 1.0 - p1_aligned,
    "class_1": p1_aligned
}, columns=["Id","class_0","class_1"])

# 5) Hard validations (fail fast with clear errors)
assert list(out.columns) == ["Id","class_0","class_1"], "Wrong column names/order"
assert len(out) == len(ssub), f"Row count {len(out)} != expected {len(ssub)}"
assert out.isna().sum().sum() == 0, "Found NaNs in submission"
assert np.isfinite(out[["class_0","class_1"]].to_numpy()).all(), "Found non-finite values"
assert ((out[["class_0","class_1"]] >= 0) & (out[["class_0","class_1"]] <= 1)).to_numpy().all(), "Values out of [0,1]"

# 6) Save with the exact file name/path Kaggle requires for Notebook submissions
out_path = "/kaggle/working/submission.csv"
out.to_csv(out_path, index=False)
print("Saved:", out_path, "| rows:", len(out))
print("Working files:", os.listdir("/kaggle/working"))
print(out.head().to_string(index=False))


