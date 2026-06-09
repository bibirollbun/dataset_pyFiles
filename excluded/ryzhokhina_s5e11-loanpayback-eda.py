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


train_path = "/kaggle/input/playground-series-s5e11/train.csv"
test_path = "/kaggle/input/playground-series-s5e11/test.csv"
submission_path = "/kaggle/input/playground-series-s5e11/sample_submission.csv"


train = pd.read_csv(train_path, index_col = 0)
print(train.shape)
train.head(10)


train.describe(include = 'all')


train.info()


# Basic constants
TARGET = 'loan_paid_back'
NUM_FEATURES = ['annual_income','debt_to_income_ratio','credit_score','loan_amount','interest_rate']
CAT_FEATURES = ['gender','marital_status','education_level','employment_status','loan_purpose','grade_subgrade']

assert TARGET in train.columns, "Target column 'loan_paid_back' not found!"


# Target distribution
print("Target distribution:")
print(train[TARGET].value_counts(normalize=True))

# Is target strictly binary (0/1)?
uniq_t = np.sort(train[TARGET].unique())
print("Unique target values:", uniq_t)

# Missing values overview
na_counts = train.isna().sum().sort_values(ascending=False)
print("Missing values per column:")
display(na_counts)


import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated")


sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.1)

# Distribution plots for numeric features with KDE overlay
for col in NUM_FEATURES:
    plt.figure(figsize=(7,4))
    sns.histplot(data=train, x=col, kde=True, bins=40, edgecolor='none', alpha=0.7)
    plt.title(f'Distribution of {col}', fontsize=14, weight='bold')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    sns.despine()
    plt.tight_layout()
    plt.show()


# Correlation with target and among numeric features
corr_df = train[NUM_FEATURES + [TARGET]].corr(numeric_only=True)

sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.1)
plt.figure(figsize=(8,6))
mask = np.triu(np.ones_like(corr_df, dtype=bool))

sns.heatmap(
    corr_df,
    mask=mask,
    cmap="viridis",
    annot=True,
    fmt=".3f",
    linewidths=0.6,
    cbar_kws={"shrink": .8, "label": "Correlation"},
    square=True
)

plt.title("Correlation Heatmap (Numeric Features + Target)", fontsize=14, weight="bold", pad=15)
plt.xticks(rotation=30, ha='right')
plt.yticks(rotation=0)
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()


n_rows = train.shape[0]
n_rows


# For each categorical feature, show top categories and a target rate by category
for col in CAT_FEATURES:
    print(f"\n==== {col} ====")
    print("Unique values:", train[col].nunique())

    order = train[col].value_counts().index
    # Plot frequency
    fig = plt.figure(figsize=(12, 4))
    
    ax1 = sns.barplot(x=train[col].value_counts().index, y=train[col].value_counts(normalize=True)*100)
    ax1.bar_label(ax1.containers[0], fmt="%.2f",label_type='edge')
    ax1.set_title(f'{col} frequency (%)', fontsize=12)
    ax1.set_xlabel(col)
    ax1.set_ylabel('% of total')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    plt.show()


print(train['grade_subgrade'].unique())
print(train['grade_subgrade'].nunique())


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


train = add_risk_score(train)
train.head(10)


# Correlation of risk_score with other numeric features ---
num_features = train.select_dtypes(include=[np.number]).columns.drop(TARGET)
corr_matrix = train[num_features].corr()[['risk_score']].sort_values(by='risk_score', ascending=False)

plt.figure(figsize=(6,6))
sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".3f", cbar=False)
plt.title('Correlation of Risk Score with Numeric Features', fontsize=13, weight='bold')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,4))
sns.barplot(x=train['risk_score'].value_counts().index, y=train['risk_score'].value_counts(normalize=True)*100)
plt.title(f'Distribution of Risk Score', fontsize=13, weight='bold')
plt.xlabel('Risk Score (1 = safest → 30 = riskiest)')
plt.ylabel('Density')
sns.despine()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,4))
mean_repay = train.groupby('risk_score')[TARGET].mean().reset_index()
sns.lineplot(data=mean_repay, x='risk_score', y=TARGET, marker='o', color='purple')
plt.title('Average Loan Paid Back by Risk Score', fontsize=13, weight='bold')
plt.xlabel('Risk Score (1 = safest → 30 = riskiest)')
plt.ylabel('Mean repayment probability')
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()


corr_spearman = train[['risk_score', TARGET]].corr(method='spearman').iloc[0,1]
print(f"Spearman correlation between risk_score and {TARGET}: {corr_spearman:.4f}")


from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import lightgbm as lgb


# === Define target and features ===
target_col = 'loan_paid_back'

# Drop redundant columns (like grade_subgrade if risk_score replaces it)
drop_cols = ['grade_subgrade'] if 'grade_subgrade' in train.columns else []
X = train.drop(columns=[target_col] + drop_cols)
y = train[target_col].astype(int)

# Identify categorical features
cat_features = [col for col in X.select_dtypes('object').columns]

# ===  Train / validation split ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# === CatBoost baseline (handles categoricals natively) ===
train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool   = Pool(X_val, y_val, cat_features=cat_features)


cat_model = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=8,
    eval_metric='AUC',
    loss_function='Logloss',
    random_state=42,
    verbose=100
)

cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True)


# === Evaluate ===
val_pred = cat_model.predict_proba(X_val)[:,1]
auc = roc_auc_score(y_val, val_pred)
print(f"\n CatBoost AUC with risk_score: {auc:.4f}")
print(classification_report(y_val, (val_pred > 0.5).astype(int)))

# === Feature importance ===
feat_imp = pd.DataFrame({
    'feature': X_train.columns,
    'importance': cat_model.get_feature_importance(train_pool)
}).sort_values('importance', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(data=feat_imp.head(15), x='importance', y='feature', palette='viridis')
plt.title('Top Feature Importances (CatBoost)')
plt.tight_layout()
plt.show()


sns.countplot(x=y_train, palette='viridis')
plt.title('Target Distribution in Training Set')
for p in plt.gca().patches:
    plt.text(p.get_x()+p.get_width()/2, p.get_height()+1000,
             f'{p.get_height()/len(y_train)*100:.1f}%', ha='center')
plt.show()


# === Add imbalansing handling
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos
print(f"scale_pos_weight = {scale_pos_weight:.3f}")


cat_model_2 = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=8,
    eval_metric='AUC',
    loss_function='Logloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    verbose=100
)

cat_model_2.fit(train_pool, eval_set=val_pool, use_best_model=True)


# === Evaluate ===
val_pred = cat_model_2.predict_proba(X_val)[:,1]
auc = roc_auc_score(y_val, val_pred)
print(f"\n CatBoost AUC with risk_score: {auc:.4f}")
print(classification_report(y_val, (val_pred > 0.5).astype(int)))

# === Feature importance ===
feat_imp = pd.DataFrame({
    'feature': X_train.columns,
    'importance': cat_model_2.get_feature_importance(train_pool)
}).sort_values('importance', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(data=feat_imp.head(15), x='importance', y='feature', palette='viridis')
plt.title('Top Feature Importances (CatBoost)')
plt.tight_layout()
plt.show()


from sklearn.preprocessing import LabelEncoder


cat_features


X_lgb = X.copy()

encoders = {}

for col in cat_features:
    le = LabelEncoder().fit(X_lgb[col].astype(str))
    X_lgb[col] = le.transform(X_lgb[col].astype(str))
    encoders[col] = le

X_train_lgb, X_val_lgb, y_train_lgb, y_val_lgb = train_test_split(
    X_lgb, y, test_size=0.2, random_state=42, stratify=y
)

neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos
print(f"scale_pos_weight = {scale_pos_weight:.3f}")


# Create LightGBM datasets
train_data = lgb.Dataset(X_train_lgb, label=y_train_lgb)
val_data   = lgb.Dataset(X_val_lgb, label=y_val_lgb)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42,
    #'is_unbalance':True
}

lgb_model = lgb.train(params, train_data, valid_sets=[val_data],
                      num_boost_round=1000)

y_pred_lgb = lgb_model.predict(X_val_lgb)
auc_lgb = roc_auc_score(y_val_lgb, y_pred_lgb)
print(f"LightGBM AUC with risk_score: {auc_lgb:.4f}")


# === Evaluate ===
y_pred_lgb = lgb_model.predict(X_val_lgb)
auc_lgb = roc_auc_score(y_val_lgb, y_pred_lgb)
print(f"LightGBM AUC with risk_score: {auc_lgb:.4f}")
print(classification_report(y_val_lgb, (y_pred_lgb > 0.5).astype(int)))

# === Feature importance ===
feat_imp = pd.DataFrame({
    'feature': X_train.columns,
    'importance': lgb_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(data=feat_imp.head(15), x='importance', y='feature', palette='viridis')
plt.title('Top Feature Importances (LightGBM)')
plt.tight_layout()
plt.show()


test = pd.read_csv(test_path, index_col = 0)
print(test.shape)
test.head()


test = add_risk_score(test)
test.head(10)


X_test = test[X.columns]


X_test.columns


for col in cat_features:
    X_test[col] = encoders[col].transform(X_test[col].astype(str))



X_test


test_pred = lgb_model.predict(X_test)


test_pred


sub = pd.read_csv(submission_path)
print(sub.shape)
sub.head()


sub.loan_paid_back = test_pred


sub.head()


sub.loan_paid_back.hist()


sub.to_csv('submission.csv', index=False)

