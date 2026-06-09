# (1) Installs 
!pip install -q lightgbm catboost

# (2) Imports & Global Setup
import warnings, gc, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler





# ============== Load ==============
DATA_DIR = "/kaggle/input/cat-in-the-dat-ii"
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
ID_COL, TARGET = "id", "target"

print("Train:", train.shape, " Test:", test.shape)



# (4) Basic cleaning shared by train/test (no leakage)
def clean_bound(df, col, lo, hi, fallback=1, as_int=True):
    if col in df.columns:
        s = pd.to_numeric(df[col], errors='coerce')
        s = s.where((s >= lo) & (s <= hi))
        fill = s.mode().iloc[0] if s.notna().any() else fallback
        s = s.fillna(fill)
        if as_int:
            s = s.round().astype(int)
        df[col] = s

for d in (train, test):
    clean_bound(d, 'month', 1, 12, fallback=1, as_int=True)
    clean_bound(d, 'day',   1, 7,  fallback=1, as_int=True)
    d.replace([np.inf, -np.inf], np.nan, inplace=True)


## EDA
sns.set_style("whitegrid")

print(f"Train shape: {train.shape} | Test shape: {test.shape}")
display(train.head())

# 1) Target distribution
plt.figure(figsize=(5,3))
sns.countplot(x='target', data=train)
plt.title('Target Distribution')
plt.show()

print("Target ratio:")
display(train['target'].value_counts(normalize=True).rename('ratio'))


# 2) Column overview by type
id_col, target_col = 'id', 'target'
feature_cols = [c for c in train.columns if c not in [id_col, target_col]]

bin_cols = [c for c in feature_cols if c.startswith('bin')]
nom_cols = [c for c in feature_cols if c.startswith('nom')]
ord_cols = [c for c in feature_cols if c.startswith('ord')]
other_cols = [c for c in feature_cols if c not in bin_cols + nom_cols + ord_cols]

print(f'Binary  ({len(bin_cols)}):', bin_cols[:10], '...')
print(f'Nominal ({len(nom_cols)}):', nom_cols[:10], '...')
print(f'Ordinal ({len(ord_cols)}):', ord_cols[:10], '...')
print(f'Other   ({len(other_cols)}):', other_cols[:10], '...')


# 3) Missing values (Top 20)
na_ratio = train[feature_cols].isna().mean().sort_values(ascending=False)
plt.figure(figsize=(8,4))
sns.barplot(x=na_ratio.values[:20], y=na_ratio.index[:20])
plt.title('Top Missing-Value Columns (train)')
plt.xlabel('NaN Ratio')
plt.tight_layout()
plt.show()


# 4) Cardinality (unique values per feature)
card = train[feature_cols].nunique(dropna=True).sort_values(ascending=False)
plt.figure(figsize=(8,4))
sns.barplot(x=card.values[:20], y=card.index[:20])
plt.title('Top High-Cardinality Columns')
plt.xlabel('Unique Values')
plt.tight_layout()
plt.show()

# Identify high-cardinality vs low-cardinality categorical features
HIGH_CARD_THRESHOLD = 20
cat_like_cols = [c for c in feature_cols if (train[c].dtype=='object') or (str(train[c].dtype)=='category')]
high_card_cols = [c for c in cat_like_cols if train[c].nunique(dropna=True) >= HIGH_CARD_THRESHOLD]
low_card_cols  = [c for c in cat_like_cols if c not in high_card_cols]
print("High-card:", high_card_cols[:10], '...')
print("Low-card :", low_card_cols[:10],  '...')


# 7) Time-based analysis (month/day)
for col, hi in [('month', 12), ('day', 7)]:
    if col in train.columns:
        plt.figure(figsize=(6,3))
        s = pd.to_numeric(train[col], errors='coerce')
        sns.barplot(x=s, y=train[target_col], estimator=np.mean, errorbar=None)
        plt.title(f'{col} vs Target Probability')
        plt.tight_layout()
        plt.show()


# 8) Correlation heatmap (after quick factorization)
X_tmp = train[feature_cols].copy()
y_tmp = train[target_col].copy()

# temporary factorization for categorical features
for c in cat_like_cols:
    X_tmp[c] = pd.factorize(X_tmp[c], sort=True)[0]

# replace NaNs with median for numeric analysis
num_cols_tmp = X_tmp.select_dtypes(include=[np.number]).columns
X_tmp[num_cols_tmp] = X_tmp[num_cols_tmp].fillna(X_tmp[num_cols_tmp].median())

corr = X_tmp.corrwith(y_tmp).sort_values(ascending=False)
top_pos = corr.head(15)
top_neg = corr.tail(15)

plt.figure(figsize=(7,5))
top_pos.plot(kind='barh'); plt.gca().invert_yaxis()
plt.title('Top Positive Correlations with Target (EDA-only)')
plt.tight_layout(); plt.show()

plt.figure(figsize=(7,5))
top_neg.plot(kind='barh')
plt.title('Top Negative Correlations with Target (EDA-only)')
plt.tight_layout(); plt.show()

print("✅ EDA complete.")



# ============== Helpers ==============
def label_fit(series):
    cats = pd.Index(series.astype("string").fillna("__NA__").unique())
    return {cat: i for i, cat in enumerate(cats)}

def label_apply(series, mapping):
    return series.astype("string").fillna("__NA__").map(mapping).fillna(-1).astype(int)

def target_encode_oof(train_col, y, val_col, n_splits=5, smoothing=20):
    gmean = y.mean()
    oof_enc = np.zeros(len(train_col))
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for tr_idx, va_idx in kf.split(train_col, y):
        tr_c, va_c = train_col.iloc[tr_idx], train_col.iloc[va_idx]
        tr_y = y.iloc[tr_idx]
        stats = tr_c.to_frame("key").join(tr_y.rename("y")).groupby("key")["y"].agg(["mean","count"])
        stats["enc"] = (stats["mean"]*stats["count"] + gmean*smoothing) / (stats["count"]+smoothing)
        oof_enc[va_idx] = va_c.map(stats["enc"]).fillna(gmean)
    stats_full = train_col.to_frame("key").join(y.rename("y")).groupby("key")["y"].agg(["mean","count"])
    stats_full["enc"] = (stats_full["mean"]*stats_full["count"] + gmean*smoothing) / (stats_full["count"]+smoothing)
    val_enc = val_col.map(stats_full["enc"]).fillna(gmean)
    return oof_enc, val_enc, stats_full["enc"]

# ============== Prepare ==============
X = train.drop(columns=[ID_COL, TARGET])
y = train[TARGET]
X_test = test.drop(columns=[ID_COL])

cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) == "category"]
high_card_cols = [c for c in cat_cols if X[c].nunique(dropna=True) >= 20]
low_card_cols  = [c for c in cat_cols if c not in high_card_cols]

# ============== Folds ==============
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_sgd = np.zeros(len(X))
te_lgb = np.zeros(len(X_test))
te_xgb = np.zeros(len(X_test))
te_sgd = np.zeros(len(X_test))
N_SPLITS = 5 
RANDOM_STATE = 42

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n========== Fold {fold}/{N_SPLITS} ==========")
    X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    X_te = X_test.copy()

    # ===== Target Encoding =====
    for c in high_card_cols:
        oof_enc, va_enc, enc_full = target_encode_oof(X_tr[c], y_tr, X_va[c], smoothing=10)
        X_tr[f"{c}_te"] = oof_enc
        X_va[f"{c}_te"] = va_enc
        X_te[f"{c}_te"] = X_te[c].map(enc_full).fillna(y_tr.mean())

    # ===== Label Encode for low-card =====
    for c in low_card_cols:
        m = label_fit(X_tr[c])
        X_tr[c] = label_apply(X_tr[c], m)
        X_va[c] = label_apply(X_va[c], m)
        X_te[c] = label_apply(X_te[c], m)

    # حذف النصوص الأصلية (بعد الترميز)
    X_tr.drop(columns=high_card_cols, inplace=True, errors="ignore")
    X_va.drop(columns=high_card_cols, inplace=True, errors="ignore")
    X_te.drop(columns=high_card_cols, inplace=True, errors="ignore")

    # ============= LightGBM =============
    dtr = lgb.Dataset(X_tr, y_tr)
    dva = lgb.Dataset(X_va, y_va)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "lambda_l2": 2.0,
        "seed": RANDOM_STATE,
        "verbosity": -1,
    }
    model_lgb = lgb.train(params, dtr, valid_sets=[dva], num_boost_round=6000,
                          callbacks=[lgb.early_stopping(300), lgb.log_evaluation(200)])
    va_lgb = model_lgb.predict(X_va)
    te_lgb += model_lgb.predict(X_te) / N_SPLITS
    oof_lgb[va_idx] = va_lgb
    print("LGB AUC:", roc_auc_score(y_va, va_lgb))



    # ============= XGBoost =============
    model_xgb = XGBClassifier(
        n_estimators=4000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False, early_stopping_rounds=300)
    va_xgb = model_xgb.predict_proba(X_va)[:,1]
    te_xgb += model_xgb.predict_proba(X_te)[:,1] / N_SPLITS
    oof_xgb[va_idx] = va_xgb
    print("XGB AUC:", roc_auc_score(y_va, va_xgb))



# ===== SGD (linear) + calibration — FIX: impute before scaling =====
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

# 1) Impute (median) للتأكد ما في NaN
imp = SimpleImputer(strategy="median")
X_tr_imp = imp.fit_transform(X_tr)   # fit على train-fold فقط
X_va_imp = imp.transform(X_va)
X_te_imp = imp.transform(X_te)

# 2) Scale
scaler = StandardScaler(with_mean=True, with_std=True)
X_tr_s = scaler.fit_transform(X_tr_imp)
X_va_s = scaler.transform(X_va_imp)
X_te_s = scaler.transform(X_te_imp)

# 3) SGD + Calibration
sgd = SGDClassifier(
    loss="log_loss",        # احتمالات
    penalty="l2",
    alpha=1e-4,
    max_iter=4000,
    tol=1e-4,
    random_state=RANDOM_STATE
)
cal = CalibratedClassifierCV(base_estimator=sgd, method="sigmoid", cv=3)
cal.fit(X_tr_s, y_tr)

va_sgd = cal.predict_proba(X_va_s)[:, 1]
te_sgd = cal.predict_proba(X_te_s)[:, 1]

oof_sgd[va_idx] = va_sgd
te_sgd += te_sgd / N_SPLITS
   
print("SGD (cal) AUC:", roc_auc_score(y_va, va_sgd))




# ============== Stacking ==============
meta_train = pd.DataFrame({"lgb": oof_lgb, "xgb": oof_xgb, "sgd": oof_sgd})
meta_test  = pd.DataFrame({"lgb": te_lgb, "xgb": te_xgb, "sgd": te_sgd})
meta_train["lgb_x_xgb"] = meta_train["lgb"] * meta_train["xgb"]
meta_test["lgb_x_xgb"]  = meta_test["lgb"] * meta_test["xgb"]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
meta_oof = np.zeros(len(meta_train))
meta_pred = np.zeros(len(meta_test))

for tr, va in skf.split(meta_train, y):
    lr = LogisticRegression(max_iter=5000)
    lr.fit(meta_train.iloc[tr], y.iloc[tr])
    meta_oof[va] = lr.predict_proba(meta_train.iloc[va])[:,1]
    meta_pred += lr.predict_proba(meta_test)[:,1] / skf.n_splits

print("Meta OOF AUC:", roc_auc_score(y, meta_oof))

# ============== Save ==============
pd.DataFrame({ID_COL: test[ID_COL], TARGET: te_lgb}).to_csv("submission_lgb.csv", index=False)
pd.DataFrame({ID_COL: test[ID_COL], TARGET: te_xgb}).to_csv("submission_xgb.csv", index=False)
pd.DataFrame({ID_COL: test[ID_COL], TARGET: te_sgd}).to_csv("submission_sgd.csv", index=False)
pd.DataFrame({ID_COL: test[ID_COL], TARGET: meta_pred}).to_csv("submission_stack.csv", index=False)

print("\n✅ Saved: submission_lgb.csv, submission_xgb.csv, submission_sgd.csv, submission_stack.csv")

