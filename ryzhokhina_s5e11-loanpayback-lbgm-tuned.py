# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import seaborn as sns
import matplotlib.pyplot as plt


train_path = "/kaggle/input/playground-series-s5e11/train.csv"
test_path = "/kaggle/input/playground-series-s5e11/test.csv"
submission_path = "/kaggle/input/playground-series-s5e11/sample_submission.csv"


train = pd.read_csv(train_path, index_col = 0)
print(train.shape)
train.head(10)


test = pd.read_csv(test_path, index_col = 0)
print(test.shape)
test.head(10)


NUM_COLS = ['annual_income','debt_to_income_ratio','credit_score','loan_amount','interest_rate']


sns.set_theme(style="whitegrid", palette="viridis")
fig, axes = plt.subplots(2, 3, figsize=(14,8))
axes = axes.ravel()

for i, col in enumerate(NUM_COLS):
    sns.boxplot(train[col], ax=axes[i])
    axes[i].set_title(col)

plt.suptitle("Boxplots — Detect Outliers in Numeric Features", fontsize=15, weight='bold')
plt.tight_layout()
plt.show()


import warnings
warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated")


fig, axes = plt.subplots(2, 3, figsize=(14,8))
axes = axes.ravel()

for i, col in enumerate(NUM_COLS):
    sns.histplot(train[col], kde=True, ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_yscale('log')  # optional to see tail better

plt.suptitle("Feature Distributions (log scale for skewed tails)", fontsize=15, weight='bold')
plt.tight_layout()
plt.show()


summary = []
for col in NUM_COLS:
    q1, q3 = train[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = max(0, q1 - 1.5 * iqr)
    upper = q3 + 1.5*iqr
    out_rate = ((train[col] < lower) | (train[col] > upper)).mean() * 100
    summary.append([col, lower, upper, out_rate])
outlier_df = pd.DataFrame(summary, columns=['Feature','LowerBound','UpperBound','%Outliers'])
display(outlier_df.round(3))


clip_stats = {
    'annual_income': (0, 110552),
    'debt_to_income_ratio': (0, 0.282),
    'loan_amount': (0, 31727),
}


def adjust_bounds(bounds, pad=0.05):
    """Expand each range by ±pad fraction (5% default)."""
    adj = {}
    for col, (low, high) in bounds.items():
        width = high - low
        adj[col] = (max(0, low - pad*width), high + pad*width)
    return adj


def clean_outliers(df, ref_bounds):
    df = df.copy()
    for col, (low, high) in ref_bounds.items():
        df[col] = df[col].clip(lower=low, upper=high)
    return df


clip_stats_adj = adjust_bounds(clip_stats, pad=0.02)


train_clean = clean_outliers(train, clip_stats_adj)
test_clean = clean_outliers(test, clip_stats_adj)


fig, axes = plt.subplots(2, 3, figsize=(14,8))
axes = axes.ravel()

for i, col in enumerate(NUM_COLS):
    sns.histplot(train_clean[col], kde=True, ax=axes[i])
    axes[i].set_title(col)

plt.suptitle("Feature Distributions (log scale for skewed tails)", fontsize=15, weight='bold')
plt.tight_layout()
plt.show()


def add_risk_score(df, column='grade_subgrade', new_col='risk_score'):
    """
    Adds a numeric risk score (1–30) to a DataFrame based on a grade_subgrade column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'grade_subgrade' column (e.g., 'A1', 'B4', 'F5').
    column : str, optional
        Name of the source column with grade/subgrade codes. Default = 'grade_subgrade'.
    new_col : str, optional
        Name of the new column to create for the numeric risk score. Default = 'risk_score'.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with an additional numeric 'risk_score' column.
    """

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    # Extract grade letter and numeric subgrade
    grade_letter = df[column].astype(str).str.extract(r'([A-F])')[0]
    subgrade_num = df[column].astype(str).str.extract(r'(\d)')[0].astype(float)

    # Map letter → numeric order
    grade_order = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
    grade_num = grade_letter.map(grade_order)

    # Compute risk score: 1–30 (A1=1 → F5=30)
    df[new_col] = ((grade_num - 1) * 5 + subgrade_num).astype(int)

    # Fill missing/invalid with -1 if any
    df[new_col] = df[new_col].fillna(-1).astype(int)

    return df


def build_base_features(df):
    out = df.copy()
    # Ratios
    out['loan_to_income'] = out['loan_amount'] / (out['annual_income'].replace(0, np.nan))
    out['loan_to_income'] = out['loan_to_income'].fillna(out['loan_to_income'].median())

    # Interactions with risk_score
    out['risk_x_interest'] = out['risk_score'] * out['interest_rate']
    out['risk_over_income_log'] = out['risk_score'] / np.log1p(out['annual_income'])

    # Monotonic buckets (quantiles)
    out['credit_bin'] = pd.qcut(out['credit_score'].rank(method='first'), q=5, labels=False)
    out['income_bin'] = pd.qcut(out['annual_income'].rank(method='first'), q=5, labels=False)

    # Clean inf/nan
    out = out.replace([np.inf, -np.inf], np.nan)
    for c in out.select_dtypes(include=[np.number]).columns:
        out[c] = out[c].fillna(out[c].median())
    return out


train_clean = add_risk_score(train_clean)
train_clean = build_base_features(train_clean)


train_clean.head(10)


test_clean = add_risk_score(test_clean)
test_clean = build_base_features(test_clean)


train_clean.columns


CAT_FEATURES = ['gender','marital_status','education_level','employment_status','loan_purpose','grade_subgrade']


from sklearn.model_selection import StratifiedKFold
def add_cv_target_encoding(train_df, test_df, cols, target, n_splits=5, smoothing=50):
    """
    KFold target encoding with smoothing:
    enc = (sum_y + global_mean * smoothing) / (count + smoothing)
    """
    df = train_df.copy()
    global_mean = df[target].mean()
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for col in cols:
        oof = pd.Series(index=df.index, dtype=float)
        for tr_idx, val_idx in kf.split(df, df[target].astype(int)):
            tr, val = df.iloc[tr_idx], df.iloc[val_idx]
            stats = tr.groupby(col)[target].agg(['sum','count'])
            enc = (stats['sum'] + global_mean * smoothing) / (stats['count'] + smoothing)
            oof.iloc[val_idx] = df.iloc[val_idx][col].map(enc).fillna(global_mean)
        df[f'{col}_te'] = oof.values

    if test_df is None:
        return df, None
    # Fit on full train and transform test
    te_maps = {}
    for col in cols:
        stats = train_df.groupby(col)[target].agg(['sum','count'])
        enc = (stats['sum'] + global_mean * smoothing) / (stats['count'] + smoothing)
        te_maps[col] = enc
    test_te = test_df.copy()
    for col in cols:
        test_te[f'{col}_te'] = test_te[col].map(te_maps[col]).fillna(global_mean)
    return df, test_te



TE_COLS = ['loan_purpose','employment_status', 'education_level'] 
TARGET_COL = 'loan_paid_back'


train_clean, test_clean = add_cv_target_encoding(train_clean, test_clean, TE_COLS, TARGET_COL )


train_clean.head(10)


test_clean.head(10)


test_clean.education_level_te.hist()
plt.show()


from sklearn.preprocessing import LabelEncoder


train_clean.info()


def label_encode(df_train, df_valid=None, df_test=None, cat_cols=None):
    encs = {}
    Xtr = df_train.copy()
    Xva = df_valid.copy() if df_valid is not None else None
    Xte = df_test.copy() if df_test is not None else None
    for c in (cat_cols or []):
        le = LabelEncoder()
        Xtr[c] = le.fit_transform(Xtr[c].astype(str))
        if Xva is not None:
            Xva[c] = Xva[c].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        if Xte is not None:
            Xte[c] = Xte[c].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        encs[c] = le
    return Xtr, Xva, Xte, encs


ALL_FEATURES = [c for c in train_clean.columns if c not in [TARGET_COL, ]]
CAT_COLS = train_clean.select_dtypes(include=['object']).columns.tolist()


print(ALL_FEATURES)


print(CAT_COLS)


import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report


tr_data, _, tr_test, encs  = label_encode(df_train = train_clean, df_test = test_clean, cat_cols = CAT_COLS)


tr_data.head()


tr_data.columns


FEATURES = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount',
       'interest_rate', 'gender', 'marital_status', 'education_level',
       'employment_status', 'loan_purpose', 'risk_score', 'loan_to_income', 'risk_x_interest',
       'risk_over_income_log', 'credit_bin', 'income_bin', 'loan_purpose_te',
       'employment_status_te', 'education_level_te']
TARGET_COL = 'loan_paid_back'


# X = tr_data[FEATURES]
# y = tr_data[TARGET_COL].astype(int)


# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# oof = np.zeros(len(train_clean))
# models = []
# fold_scores = []

# for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
#     X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
#     y_tr, y_va = y[tr_idx], y[va_idx]

#     # imbalance weight per fold
#     neg, pos = np.bincount(y_tr)
#     spw = neg / max(pos, 1)

#     train_ds = lgb.Dataset(X_tr, label=y_tr)
#     valid_ds = lgb.Dataset(X_va, label=y_va)

#     params = {
#         'objective': 'binary',
#         'metric': ['auc','aucpr'],
#         'learning_rate': 0.03,
#         'num_leaves': 96,
#         'max_depth': -1,
#         'min_data_in_leaf': 60,
#         'feature_fraction': 0.85,
#         'bagging_fraction': 0.85,
#         'bagging_freq': 5,
#         'lambda_l1': 0.0,
#         'lambda_l2': 2.0,
#         'scale_pos_weight': spw,     # imbalance handling (don’t also set is_unbalance)
#         'seed': 42,
#         'verbosity': -1
#     }
#     model = lgb.train(
#         params, train_ds,
#         valid_sets=[valid_ds],
#         num_boost_round=1000
#     )
#     pred = model.predict(X_va, num_iteration=model.best_iteration)
#     score = roc_auc_score(y_va, pred)
#     oof[va_idx] = pred
#     fold_scores.append(score)
#     models.append(model)
#     print(f'Fold {fold} AUC: {score:.5f}')

# oof_auc = roc_auc_score(tr_data[TARGET_COL], oof)
# print(f'\nOOF AUC (5-fold): {oof_auc:.5f} | folds: {[round(s,5) for s in fold_scores]}')



# # === Feature importance ===
# feat_imp = pd.DataFrame({
#     'feature': FEATURES,
#     'importance': models[1].feature_importance(importance_type='gain')
# }).sort_values('importance', ascending=False)

# plt.figure(figsize=(8,5))
# sns.barplot(data=feat_imp.head(15), x='importance', y='feature', palette='viridis')
# plt.title('Top Feature Importances (LightGBM)')
# plt.tight_layout()
# plt.show()


tr_data.head()


FEATURES = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount',
       'interest_rate', 'gender', 'marital_status', 
       'risk_score', 'loan_to_income', 'risk_x_interest',
       'risk_over_income_log', 'credit_bin', 'income_bin', 'loan_purpose_te',
       'employment_status_te', 'education_level']
TARGET_COL = 'loan_paid_back'


len(FEATURES)


X = tr_data[FEATURES]
y = tr_data[TARGET_COL].astype(int)
X_test = tr_test[FEATURES]


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train_clean))
test_pred = np.zeros(len(X_test))
models = []
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # imbalance weight per fold
    neg, pos = np.bincount(y_tr)
    spw = neg / max(pos, 1)

    train_ds = lgb.Dataset(X_tr, label=y_tr)
    valid_ds = lgb.Dataset(X_va, label=y_va)

    params = {
        'objective': 'binary',
        'metric': ['auc','aucpr'],
        'learning_rate': 0.03,
        'num_leaves': 96,
        'max_depth': -1,
        'min_data_in_leaf': 60,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'lambda_l1': 0.0,
        'lambda_l2': 2.0,
        'scale_pos_weight': spw,     
        'seed': 42,
        'verbosity': -1
    }
    model = lgb.train(
        params, train_ds,
        valid_sets=[valid_ds],
        num_boost_round=1000
    )
    pred = model.predict(X_va, num_iteration=model.best_iteration)
        # Test preds (accumulate, then average after loop)
    test_pred += model.predict(X_test, num_iteration=model.best_iteration)
    
    score = roc_auc_score(y_va, pred)
    oof[va_idx] = pred
    fold_scores.append(score)
    models.append(model)
    print(f'Fold {fold} AUC: {score:.5f}')

oof_auc = roc_auc_score(tr_data[TARGET_COL], oof)
print(f'\nOOF AUC (5-fold): {oof_auc:.5f} | folds: {[round(s,5) for s in fold_scores]}')

# Average test predictions over folds
test_pred /= kf.get_n_splits()


# === Feature importance ===
feat_imp = pd.DataFrame({
    'feature': FEATURES,
    'importance': models[1].feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(data=feat_imp.head(15), x='importance', y='feature', palette='viridis')
plt.title('Top Feature Importances (LightGBM)')
plt.tight_layout()
plt.show()


sub = pd.read_csv(submission_path)
print(sub.shape)
sub.head()


sub.loan_paid_back = test_pred
sub.head(10)


sub.to_csv('submission.csv', index=False)

