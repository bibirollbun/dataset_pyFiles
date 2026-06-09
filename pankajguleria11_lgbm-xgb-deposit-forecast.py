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


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

try:
    from sklearn.preprocessing import TargetEncoder  
    SKLEARN_TE = True
except Exception:
    from category_encoders import TargetEncoder  
    SKLEARN_TE = False


TARGET = 'y'
ID_COL = 'id'
USE_DURATION = True
N_SPLITS = 5
RANDOM_STATE = 42
VERBOSE_EVAL = 100

train = pd.read_csv(r'/kaggle/input/playground-series-s5e8/train.csv')
test  = pd.read_csv(r'/kaggle/input/playground-series-s5e8/test.csv')


train[TARGET] = train[TARGET].astype(int)
train.head(2)


# Defining a function to convert any values in a column if their respective frequency is less than 100 to __rare__
def cap_rare_categories(s: pd.Series, min_count=100):
    counts = s.value_counts()
    rare = counts[counts < min_count].index
    return s.where(~s.isin(rare), other="__rare__")


df = train.copy()
te_df = test.copy()

# Cap rare categories
for c in ['job','education','poutcome']:
    if c in df.columns:
        df[c] = cap_rare_categories(df[c], min_count=100)
    if c in te_df.columns:
        te_df[c] = cap_rare_categories(te_df[c], min_count=100)

# Month to ordinal
month_order = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
if 'month' in df.columns:
    df['month'] = df['month'].astype(str).str.lower()
    te_df['month'] = te_df['month'].astype(str).str.lower()
    df['month_ord'] = pd.Categorical(df['month'], categories=month_order, ordered=True).codes
    te_df['month_ord'] = pd.Categorical(te_df['month'], categories=month_order, ordered=True).codes

# Binary map for yes/no
yn_map = {'yes': 1, 'no': 0}
for c in ['default','housing','loan']:
    if c in df.columns:
        df[c] = df[c].map(yn_map)
    if c in te_df.columns:
        te_df[c] = te_df[c].map(yn_map)

# Contact history flags
if set(['previous','pdays']).issubset(df.columns):
    df['was_contacted_before'] = (df['pdays'] >= 0).astype(int)
    te_df['was_contacted_before'] = (te_df['pdays'] >= 0).astype(int)

# Interactions (duration)
if set(['duration','campaign']).issubset(df.columns):
    df['dur_per_contact'] = df['duration'] / np.clip(df['campaign'], 1, None)
    te_df['dur_per_contact'] = te_df['duration'] / np.clip(te_df['campaign'], 1, None)

# Converting age into bins
if 'age' in df.columns:
    age_bins = [0,25,35,45,55,65,120]
    df['age_bin'] = pd.cut(df['age'], bins=age_bins, right=True)
    te_df['age_bin'] = pd.cut(te_df['age'], bins=age_bins, right=True)

# Additional engineered features in contacted before (pdays), poutcome, balance and campaign
def add_more_features(frame: pd.DataFrame):
    f = frame
    if 'pdays' in f.columns:
        f['was_never_contacted'] = (f['pdays'] == -1).astype(int)
        f['recent_contact_7d'] = ((f['pdays'] >= 0) & (f['pdays'] <= 7)).astype(int)
        f['recent_contact_30d'] = ((f['pdays'] >= 0) & (f['pdays'] <= 30)).astype(int)
        f['recent_contact_90d'] = ((f['pdays'] >= 0) & (f['pdays'] <= 90)).astype(int)
        f['pdays_nonneg'] = np.where(f['pdays']>=0, f['pdays'], np.nan)
        f['pdays_bucket'] = pd.cut(f['pdays_nonneg'], bins=[-0.1,7,30,90,180,9999])
    if 'poutcome' in f.columns:
        f['prev_success'] = (f['poutcome'].astype(str).str.lower()=='success').astype(int)
    if 'balance' in f.columns:
        f['log1p_balance'] = np.log1p(np.clip(f['balance'], a_min=0, a_max=None))
    if 'campaign' in f.columns:
        f['high_campaign'] = (f['campaign'] > 3).astype(int)
    return f

df = add_more_features(df)
te_df = add_more_features(te_df)


# Optionally drop duration-based leakage
drop_cols = []
if not USE_DURATION:
    for c in ['duration', 'dur_per_contact']:
        if c in df.columns:
            drop_cols.append(c)


# Preparing features
features_all = [c for c in df.columns if c not in [TARGET] + drop_cols]
X = df[features_all].copy()
y = df[TARGET].values
X_test = te_df[features_all].copy()

# Identifing categorical columns
tree_cat_cols = [c for c in X.columns if (X[c].dtype == 'object') or (str(X[c].dtype).startswith('category'))]
for c in ['age_bin', 'pdays_bucket']:
    if c in X.columns and c not in tree_cat_cols:
        tree_cat_cols.append(c)

# Determining numeric columns
num_cols_all = [c for c in X.columns if c not in tree_cat_cols]


# Defining a function for bar plot of target rate by feature
def plot_target_rate_bar(df_in, feature, target=TARGET, top_k=None, order=None):
    if feature not in df_in.columns:
        return
    tmp = df_in[[feature, target]].copy()
    if pd.api.types.is_interval_dtype(tmp[feature]) or pd.api.types.is_categorical_dtype(tmp[feature]):
        tmp[feature] = tmp[feature].astype(str)
    grp = tmp.groupby(feature)[target].mean().reset_index().rename(columns={target: "target_rate"})
    if top_k is not None and order is None:
        counts = tmp[feature].value_counts().reset_index()
        counts.columns = [feature, "n"]
        grp = grp.merge(counts, on=feature, how="left").sort_values("n", ascending=False).head(top_k)
    if order is not None:
        grp = grp.set_index(feature).reindex(order).reset_index()
    plt.figure(figsize=(8,4))
    ax = sns.barplot(data=grp, x=feature, y="target_rate", color="#4472c4")
    ax.set_title(f"Target rate by {feature}")
    ax.set_ylabel("Target rate")
    ax.set_xlabel(feature)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


# Defining a function for numeric feature distribution with target overlay
def plot_dist_with_target(df_in, feature, target=TARGET, log=False, bins=50):
    if feature not in df_in.columns:
        return
    s = df_in[feature].copy()
    if log:
        s = np.log1p(np.clip(s, a_min=0, a_max=None))
    plt.figure(figsize=(8,4))
    try:
        sns.kdeplot(data=df_in, x=s, hue=df_in[target].astype(str), fill=True, common_norm=False, alpha=0.4)
    except Exception:
        sns.histplot(s, bins=bins, kde=True, color="#4472c4", alpha=0.6)
    ttl = f"Distribution of {feature}" + (" (log1p)" if log else "")
    plt.title(ttl)
    plt.xlabel(feature)
    plt.tight_layout()
    plt.show()


# Target rate by month (seasonality)
if 'month_ord' in X.columns:
    # Order from 0..11 if present
    order = sorted(X['month_ord'].dropna().unique().tolist())
    plot_target_rate_bar(df.assign(month_ord=X['month_ord']), 'month_ord')


# Distribution of duration with target overlay
if 'duration' in X.columns:
    plot_dist_with_target(df, 'duration', target=TARGET, log=False)


# Distribution of duration per contact
if 'dur_per_contact' in X.columns:
    plot_dist_with_target(df, 'dur_per_contact', target=TARGET, log=True)


# Target rate by pdays-derived features
for feat in ['was_contacted_before','was_never_contacted','recent_contact_7d','recent_contact_30d','recent_contact_90d']:
    if feat in X.columns:
        plot_target_rate_bar(df, feat, target=TARGET)


# Target rate by pdays bucket
if 'pdays_bucket' in X.columns:
    plot_target_rate_bar(df, 'pdays_bucket', target=TARGET)


# Target rate by age bin
if 'age_bin' in X.columns:
    plot_target_rate_bar(df, 'age_bin', target=TARGET)


# Target rate by balance
if 'log1p_balance' in X.columns:
    plot_dist_with_target(df, 'log1p_balance', target=TARGET, log=False)


# Target rate by campaign intensity
if 'high_campaign' in X.columns:
    plot_target_rate_bar(df, 'high_campaign', target=TARGET)


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# LightGBM params
lgb_base_params = dict(
    objective='binary',
    boosting_type='gbdt',
    learning_rate=0.03,
    n_estimators=6000,
    num_leaves=48,
    max_depth=-1,
    min_data_in_leaf=60,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    lambda_l1=0.0,
    lambda_l2=8.0,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# XGBoost params
xgb_base_params = dict(
    objective='binary:logistic',
    eval_metric='auc',
    learning_rate=0.03,
    n_estimators=6000,
    max_depth=6,
    min_child_weight=3.0,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.0,
    reg_lambda=8.0,
    tree_method='hist',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    early_stopping_rounds=200  # moved here
)


lgb_oof = np.zeros(len(X), dtype=float)
xgb_oof = np.zeros(len(X), dtype=float)
lgb_models = []
xgb_models = []

def build_target_encoder():
    if SKLEARN_TE:
        return TargetEncoder(cv=5, smooth='auto', random_state=RANDOM_STATE)
    else:
        return TargetEncoder(min_samples_leaf=20, smoothing=10)

def ensure_df(te_out, index, col_names):
    if isinstance(te_out, pd.DataFrame):
        out = te_out.copy()
        out.index = index
        out.columns = col_names
        return out
    else:
        return pd.DataFrame(te_out, index=index, columns=col_names)


for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n===== Fold {fold}/{N_SPLITS} =====")
    X_tr_raw, y_tr = X.iloc[tr_idx].copy(), y[tr_idx]
    X_va_raw, y_va = X.iloc[va_idx].copy(), y[va_idx]

    # Imputing numeric columns
    num_imputer = SimpleImputer(strategy='median')
    if len(num_cols_all) > 0:
        X_tr_raw[num_cols_all] = num_imputer.fit_transform(X_tr_raw[num_cols_all])
        X_va_raw[num_cols_all] = num_imputer.transform(X_va_raw[num_cols_all])

    # Target Encoding for categoricals with cross-fitting on train, then transform on valid
    te_enc = build_target_encoder()
    if len(tree_cat_cols) > 0:
        X_tr_cats = X_tr_raw[tree_cat_cols].astype(str)
        X_va_cats = X_va_raw[tree_cat_cols].astype(str)

        if SKLEARN_TE:
            X_tr_te = te_enc.fit_transform(X_tr_cats, y_tr)
            X_va_te = te_enc.transform(X_va_cats)
        else:
            te_enc.fit(X_tr_cats, y_tr)
            X_tr_te = te_enc.transform(X_tr_cats)
            X_va_te = te_enc.transform(X_va_cats)

        # Ensuring DataFrame outputs with explicit column names
        te_cols = [f"{c}_te" for c in tree_cat_cols]
        X_tr_te = ensure_df(X_tr_te, X_tr_cats.index, te_cols)
        X_va_te = ensure_df(X_va_te, X_va_cats.index, te_cols)

        # Replacing categorical columns with TE features
        X_tr_fe = pd.concat([X_tr_raw.drop(columns=tree_cat_cols).reset_index(drop=True),
                             X_tr_te.reset_index(drop=True)], axis=1)
        X_va_fe = pd.concat([X_va_raw.drop(columns=tree_cat_cols).reset_index(drop=True),
                             X_va_te.reset_index(drop=True)], axis=1)
    else:
        X_tr_fe, X_va_fe = X_tr_raw.copy(), X_va_raw.copy()

    fold_cols = X_tr_fe.columns.tolist()
    X_tr_np = X_tr_fe.to_numpy(dtype=np.float32)
    X_va_np = X_va_fe.to_numpy(dtype=np.float32)

    pos = y_tr.sum()
    neg = len(y_tr) - pos
    spw = (neg / pos) if pos > 0 else 1.0

    # LightGBM
    lgbm = LGBMClassifier(**{**lgb_base_params, **{'scale_pos_weight': spw}})
    lgbm.fit(
        X_tr_np, y_tr,
        eval_set=[(X_va_np, y_va)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(VERBOSE_EVAL)]
    )
    lgb_models.append((lgbm, num_imputer, te_enc, fold_cols))
    lgb_oof[va_idx] = lgbm.predict_proba(X_va_np)[:, 1]

    # XGBoost
    xgbm = XGBClassifier(**{**xgb_base_params, **{'scale_pos_weight': spw}})
    xgbm.fit(
    X_tr_np, y_tr,
    eval_set=[(X_va_np, y_va)],
    verbose=False
    )
    xgb_models.append((xgbm, num_imputer, te_enc, fold_cols))
    xgb_oof[va_idx] = xgbm.predict_proba(X_va_np)[:, 1]

# Reporting AUCs
lgb_auc = roc_auc_score(y, lgb_oof)
xgb_auc = roc_auc_score(y, xgb_oof)
print(f"\nLightGBM OOF ROC-AUC: {lgb_auc:.6f}")
print(f"XGBoost  OOF ROC-AUC: {xgb_auc:.6f}")


# Imputing and reporting blended AUCs
weights = np.linspace(0.2, 0.8, 13)
best_auc, best_w = -1.0, None
for w in weights:
    blend = w * lgb_oof + (1.0 - w) * xgb_oof
    auc = roc_auc_score(y, blend)
    if auc > best_auc:
        best_auc, best_w = auc, w
print(f"Blended OOF ROC-AUC: {best_auc:.6f} with LGB weight={best_w:.2f}")


# Prepare test data to match training fold schema

def prepare_test_for_fold(Xt: pd.DataFrame, num_imputer, te_enc, cat_cols, num_cols_all, fold_cols):
    Xt = Xt.copy()

    # Imputing numerics
    if len(num_cols_all) > 0:
        Xt[num_cols_all] = num_imputer.transform(Xt[num_cols_all])

    # Target encoding for categoricals
    if len(cat_cols) > 0:
        Xt_cats = Xt[cat_cols].astype(str)
        Xt_te = te_enc.transform(Xt_cats)

        te_cols = [f"{c}_te" for c in cat_cols]
        if isinstance(Xt_te, pd.DataFrame):
            Xt_te = Xt_te.copy()
            Xt_te.index = Xt_cats.index
            Xt_te.columns = te_cols
        else:
            Xt_te = pd.DataFrame(Xt_te, index=Xt_cats.index, columns=te_cols)

        Xt_num = Xt.drop(columns=cat_cols)
        Xt = pd.concat([Xt_num.reset_index(drop=True), Xt_te.reset_index(drop=True)], axis=1)

    # Align test features with training fold columns
    missing_cols = [c for c in fold_cols if c not in Xt.columns]
    for c in missing_cols:
        Xt[c] = 0.0
    extra_cols = [c for c in Xt.columns if c not in fold_cols]
    if extra_cols:
        Xt = Xt.drop(columns=extra_cols)

    Xt = Xt[fold_cols]
    return Xt.to_numpy(dtype=np.float32)


# Generating test predictions with LGB & XGB (blended across folds)

lgb_fold_preds = []
for (lgbm, num_imp, te_enc, fold_cols) in lgb_models:
    Xt_np = prepare_test_for_fold(X_test, num_imp, te_enc, tree_cat_cols, num_cols_all, fold_cols)
    preds = lgbm.predict_proba(Xt_np)[:, 1]
    lgb_fold_preds.append(preds)
lgb_test_pred = np.mean(lgb_fold_preds, axis=0)

xgb_fold_preds = []
for (xgbm, num_imp, te_enc, fold_cols) in xgb_models:
    Xt_np = prepare_test_for_fold(X_test, num_imp, te_enc, tree_cat_cols, num_cols_all, fold_cols)
    preds = xgbm.predict_proba(Xt_np)[:, 1]
    xgb_fold_preds.append(preds)
xgb_test_pred = np.mean(xgb_fold_preds, axis=0)


# Final blended test predictions
if best_w is None:
    best_w = 0.5
test_blend = best_w * lgb_test_pred + (1.0 - best_w) * xgb_test_pred


if ID_COL not in test.columns:
    raise ValueError(f"ID column '{ID_COL}' not found in test data. Please set ID_COL correctly.")

submission = pd.DataFrame({
    ID_COL: test[ID_COL].values,
    'y_pred_proba': test_blend.astype(float)
})
submission['y'] = submission['y_pred_proba'].round(9)
submission = submission[['id','y']]

submission.to_csv('/kaggle/working/final_submission.csv', index=False)

print("Predictions saved as final_submission.csv")




