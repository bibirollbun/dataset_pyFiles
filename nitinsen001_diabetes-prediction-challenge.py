# Requisitos:
# pip install pandas numpy scikit-learn lightgbm catboost category_encoders joblib shap
# (En Kaggle la mayoría ya están instalados)

import os
import glob
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import category_encoders as ce
import joblib
import lightgbm as lgb
from catboost import CatBoostClassifier

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------
# 1) Robust file loading
# ---------------------------
def find_paths(preferred_names=("train.csv","test.csv")):
    candidates = []
    # Current dir
    for name in preferred_names:
        p = os.path.join(".", name)
        if os.path.exists(p):
            candidates.append(p)
    # Kaggle competition pattern
    kaggle_pattern = "/kaggle/input/*"
    for base in glob.glob(kaggle_pattern):
        for name in preferred_names:
            p = os.path.join(base, name)
            if os.path.exists(p):
                candidates.append(p)
    return list(dict.fromkeys(candidates))

paths = find_paths()
if len(paths) >= 2:
    TRAIN_PATH = [p for p in paths if p.endswith("train.csv")][0]
    TEST_PATH  = [p for p in paths if p.endswith("test.csv")][0]
else:
    # If running in Colab, allow upload
    try:
        from google.colab import files
        print("No train/test found automatically. Use the upload widget (Colab).")
        uploaded = files.upload()  # user must upload train.csv and test.csv
        uploaded_files = list(uploaded.keys())
        TRAIN_PATH = [f for f in uploaded_files if "train" in f.lower()][0]
        TEST_PATH  = [f for f in uploaded_files if "test" in f.lower()][0]
    except Exception:
        csvs = glob.glob("*.csv")
        print("CSV files in current folder:", csvs)
        if "train.csv" in csvs and "test.csv" in csvs:
            TRAIN_PATH = "train.csv"
            TEST_PATH = "test.csv"
        else:
            raise FileNotFoundError(
                "No train/test CSV found. Put 'train.csv' and 'test.csv' in your working directory "
                "or run this in a Kaggle notebook (where files are in /kaggle/input/...)."
            )

print("Using:")
print("TRAIN_PATH =", TRAIN_PATH)
print("TEST_PATH  =", TEST_PATH)

# ---------------------------
# 2) Load data (expected columns: id, diagnosed_diabetes in train)
# ---------------------------
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

# Validate expected columns
if 'id' not in train.columns:
    raise KeyError("train.csv must contain column 'id'")
if 'diagnosed_diabetes' not in train.columns:
    raise KeyError("train.csv must contain column 'diagnosed_diabetes' (0/1)")
if 'id' not in test.columns:
    raise KeyError("test.csv must contain column 'id'")

# Basic targets and ids
y = train['diagnosed_diabetes'].astype(int).reset_index(drop=True)
train_id = train['id']
test_id  = test['id']

# Drop id + target from features
X = train.drop(columns=['id','diagnosed_diabetes'])
X_test = test.drop(columns=['id'])

# ---------------------------
# 3) Simple EDA (quick checks)
# ---------------------------
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Target distribution (proportions):")
print(y.value_counts(normalize=True))

# ---------------------------
# 4) Feature engineering (safe, generic)
# ---------------------------
def safe_feature_engineering(df):
    df = df.copy()
    # count nulls
    df['n_nulls'] = df.isnull().sum(axis=1)
    # numeric aggregates if exist
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 0:
        df['num_mean'] = df[num_cols].mean(axis=1)
        df['num_std']  = df[num_cols].std(axis=1).fillna(0)
        df['num_sum'] = df[num_cols].sum(axis=1)
    # convert obvious date-like columns to epoch if present
    for c in df.columns:
        if ('date' in c.lower()) or ('fecha' in c.lower()):
            df[c] = pd.to_datetime(df[c], errors='coerce')
            try:
                df[c] = df[c].view('int64') // 10**9
            except Exception:
                df[c] = pd.to_numeric(df[c], errors='coerce')
    # cap extremely large values to reduce outliers effect (1st/99th percentile)
    for c in df.select_dtypes(include=[np.number]).columns:
        series = df[c].dropna()
        if len(series) > 0:
            low = np.nanpercentile(series, 1)
            high = np.nanpercentile(series, 99)
            df[c] = df[c].clip(lower=low, upper=high)
    return df

X = safe_feature_engineering(X)
X_test = safe_feature_engineering(X_test)

# ---------------------------
# 5) Identify categorical vs numeric
# ---------------------------
cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]
print("Numeric cols:", len(num_cols), "Categorical cols:", len(cat_cols))

# ---------------------------
# 6) Encoding and preprocessing (target encoder + imputing + scaling)
# ---------------------------
# Target encoder for categorical cols (uses smoothing)
if len(cat_cols) > 0:
    te = ce.TargetEncoder(cols=cat_cols, smoothing=0.3)
    X_enc = te.fit_transform(X, y)
    X_test_enc = te.transform(X_test)
else:
    te = None
    X_enc = X.copy(); X_test_enc = X_test.copy()

# Impute numeric
imputer = SimpleImputer(strategy='median')
if len(num_cols) > 0:
    X_enc[num_cols] = imputer.fit_transform(X_enc[num_cols])
    X_test_enc[num_cols] = imputer.transform(X_test_enc[num_cols])

# Scale numeric
scaler = StandardScaler()
if len(num_cols) > 0:
    X_enc[num_cols] = scaler.fit_transform(X_enc[num_cols])
    X_test_enc[num_cols] = scaler.transform(X_test_enc[num_cols])

# Save preprocessors
joblib.dump(imputer, "imputer.joblib")
joblib.dump(scaler, "scaler.joblib")
if te:
    joblib.dump(te, "target_encoder.joblib")

# ---------------------------
# 7) Modeling: CV, OOF preds for stacking
# ---------------------------
NFOLDS = 5
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_SEED)

oof_lgb = np.zeros(len(X_enc))
oof_cat = np.zeros(len(X_enc))
preds_lgb = np.zeros(len(X_test_enc))
preds_cat = np.zeros(len(X_test_enc))

lgb_params = {
    'objective':'binary',
    'metric':'auc',
    'verbosity': -1,
    'boosting_type':'gbdt',
    'learning_rate':0.05,
    'num_leaves': 31,
    'seed':RANDOM_SEED,
    'n_jobs':-1,
    'feature_fraction':0.8,
    'bagging_fraction':0.8,
    'bagging_freq':5
}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_enc, y)):
    print(f"Fold {fold+1}/{NFOLDS}")
    X_tr, X_val = X_enc.iloc[tr_idx], X_enc.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # ---------- LightGBM (new callback style compatible con Kaggle) ----------
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val)
    lgb_model = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200)
        ],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    preds_lgb += lgb_model.predict(X_test_enc, num_iteration=lgb_model.best_iteration) / NFOLDS

    # ---------- CatBoost ----------
    cat_model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        eval_metric='AUC',
        random_seed=RANDOM_SEED,
        early_stopping_rounds=100,
        verbose=0
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:,1]
    preds_cat += cat_model.predict_proba(X_test_enc)[:,1] / NFOLDS

# OOF scores
print("OOF LGB AUC:", roc_auc_score(y, oof_lgb))
print("OOF CAT AUC:", roc_auc_score(y, oof_cat))

# ---------------------------
# 8) Stack (meta-model) and calibration
# ---------------------------
stack_train = np.vstack([oof_lgb, oof_cat]).T
stack_test  = np.vstack([preds_lgb, preds_cat]).T

# Meta model (Logistic)
meta = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
meta.fit(stack_train, y)
stack_oof = meta.predict_proba(stack_train)[:,1]
print("Raw Stack OOF AUC (before calibration):", roc_auc_score(y, stack_oof))

# Calibrate meta with CalibratedClassifierCV (isotonic recommended if enough data)
calibrator = CalibratedClassifierCV(base_estimator=meta, cv=5, method='isotonic')
calibrator.fit(stack_train, y)
final_preds = calibrator.predict_proba(stack_test)[:,1]

# ---------------------------
# 9) Save models and produce submission
# ---------------------------
joblib.dump(lgb_model, "last_lgb_model.joblib")
joblib.dump(cat_model, "last_cat_model.joblib")
joblib.dump(meta, "meta_logreg.joblib")
joblib.dump(calibrator, "meta_calibrator.joblib")

submission = pd.DataFrame({'id': test_id, 'diagnosed_diabetes': final_preds})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv (id,diagnosed_diabetes).")

# ---------------------------
# 10) Quick optional diagnostics (feature importance)
# ---------------------------
try:
    import matplotlib.pyplot as plt
    # LightGBM feature importance (last fold)
    fig, ax = plt.subplots(figsize=(6,10))
    lgb.plot_importance(lgb_model, max_num_features=30, ax=ax)
    plt.tight_layout()
    plt.show()
except Exception:
    pass

print("Done. Next steps for improved leaderboard performance:")
print("- Add more feature engineering (temporal aggregations, domain features).")
print("- Add XGBoost and additional seeds / bagging to reduce variance.")
print("- Run hyperparameter tuning (Optuna).")
print("- If there is temporal structure, use time-based CV instead of StratifiedKFold.")


