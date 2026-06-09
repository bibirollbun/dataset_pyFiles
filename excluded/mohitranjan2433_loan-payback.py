import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import category_encoders as ce

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
np.set_printoptions(suppress=True)


df_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


df_train.shape, df_test.shape


df_train.info()


df_train.describe()


df_train[df_train['annual_income'] == df_train['annual_income'].max()][['annual_income','loan_amount', 'interest_rate','debt_to_income_ratio','loan_paid_back','credit_score']]


df_train[df_train['annual_income'] == df_train['annual_income'].min()][['annual_income','loan_amount', 'interest_rate','debt_to_income_ratio','loan_paid_back']]


df_train.isnull().sum()


column_continuos = [
    'annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate'
]

column_categorical = [
    'gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade', 'grade_subgrade'
]


num_plots = len(column_continuos)
num_cols = 2
num_rows = (num_plots + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(16,5 * num_rows))
axes = axes.flatten()

for i, col in enumerate(column_continuos):
    sns.boxplot(x=df_train[col], ax=axes[i])
    axes[i].set_title(col)

for j in range(i + 1, num_rows * num_cols):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


plt.scatter(df_train['loan_amount'], df_train['debt_to_income_ratio'],
            alpha=0.2, s=25)

plt.xlabel('Loan Amount ($)')
plt.ylabel('Debt-to-Income Ratio')
plt.title('Debt to Income Ratio vs Loan Amount')

sns.despine(trim=True, left=True)
plt.show()


plt.figure(figsize=(16, 8))
plt.scatter(df_train['annual_income'], df_train['credit_score'], alpha=0.3)
plt.xlabel('Annual Income')
plt.ylabel('Credit Score')
plt.title('Credit Score vs Annual Income')
plt.show()


df_train.info()


sns.violinplot(data=df_train, x='loan_paid_back', y='credit_score')
plt.title("Credit Score vs Loan Paid Back")
plt.xlabel("Loan Paid Back")
plt.ylabel("Credit Score")
plt.show()


sns.violinplot(data=df_train, x='loan_paid_back', y='interest_rate')
plt.title("Interest_rate vs Loan Paid Back")
plt.xlabel("Loan Paid Back")
plt.ylabel("Interest_rate")
plt.show()


sns.violinplot(data=df_train, x='loan_paid_back', y='debt_to_income_ratio')
plt.title("Debt_to_income_ratio vs Loan Paid Back")
plt.xlabel("Loan Paid Back")
plt.ylabel("Debt_to_income_ratio")
plt.show()


df_low_dti_defaulters = df_train[
    (df_train['loan_paid_back'] == 0.0) &
    (df_train['debt_to_income_ratio'] < 0.05)
]


df_low_dti_defaulters.shape


df_low_dti_defaulters['annual_income'].describe()


sns.histplot(df_low_dti_defaulters['annual_income'], kde=True)
plt.title("Annual Income Distribution (Low DTI Defaulters)")
plt.show()


df_low_dti_defaulters['credit_score'].describe()


df_low_dti_defaulters['loan_purpose'].value_counts()


df_low_dti_defaulters['loan_purpose'].value_counts().plot(kind='bar')
plt.title("Loan Purpose for Low DTI Defaulters")
plt.xticks(rotation=45)
plt.ylabel("Count")
plt.show()


df_low_dti_defaulters[['annual_income', 'credit_score', 'loan_purpose']]\
    .groupby('loan_purpose')\
    .agg(['count', 'mean', 'median'])


plt.figure(figsize=(20, 25))

categorical_cols = [
    'gender', 'marital_status', 'education_level',
    'employment_status', 'loan_purpose'
]

for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 2, i)
    sns.countplot(data=df_train, x=col, hue='loan_paid_back')
    plt.title(f"{col} vs Loan Paid Back")
    plt.xticks(rotation=45)
    plt.ylabel("Count")

plt.tight_layout()
plt.show()


default_rate = df_train.groupby('grade_subgrade').loan_paid_back.mean()
default_rate





plt.figure(figsize=(16, 6))
sns.barplot(
    data=df_train,
    x='grade_subgrade',
    y='loan_paid_back',
    palette='viridis'
)
plt.title("Default Rate by Grade Subgrade (Ordered)")
plt.xlabel("Grade Subgrade")
plt.ylabel("Default Rate")
plt.xticks(rotation=90)
plt.show()


df_train['grade'] = df_train['grade_subgrade'].str[0]
df_train['sub'] = df_train['grade_subgrade'].str[1].astype(int)


sorted_order = df_train.sort_values(['grade', 'sub'])['grade_subgrade'].unique()


plt.figure(figsize=(16, 6))
sns.barplot(
    data=df_train,
    x='grade_subgrade',
    y='loan_paid_back',
    order=sorted_order,
    palette='viridis'
)
plt.title("Default Rate by Grade Subgrade (Ordered)")
plt.xlabel("Grade Subgrade")
plt.ylabel("Default Rate")
plt.xticks(rotation=90)
plt.show()


df_train.info()


df_train['loan_to_income'] = df_train['loan_amount'] / df_train['annual_income']


df_train.loan_to_income.describe()


bins = [300, 580, 670, 740, 800, 851]   
labels = ["Poor", "Fair", "Good", "Very Good", "Exceptional"]

df_train['credit_score_bucket'] = pd.cut(
    df_train['credit_score'],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=False
)


df_train['credit_score_bucket'] = df_train['credit_score_bucket'].astype(
    pd.CategoricalDtype(categories=labels, ordered=True)
)


df_train[['credit_score', 'credit_score_bucket']].head()


df_train['credit_score_bucket'].value_counts().sort_index()


sns.barplot(
    data=df_train,
    x='credit_score_bucket',
    y=1 - df_train['loan_paid_back'],  # default rate
    estimator=np.mean,
    palette='magma'
)
plt.title("Default Rate by Credit Score Bucket")
plt.xlabel("Credit Score Category")
plt.ylabel("Default Rate")
plt.show()


bins = [0.01, 0.05, 0.25, float('inf')]
labels = [
    "Low",          
    "Moderate",           
    "High"                  
]

df_train['DTI_Risk_Bucket'] = pd.cut(
    df_train['debt_to_income_ratio'],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=False
)


df_train['DTI_Risk_Bucket'] = df_train['DTI_Risk_Bucket'].astype(
    pd.CategoricalDtype(categories=labels, ordered=True)
)


df_train['DTI_Risk_Bucket'].value_counts().sort_index()


df_train.groupby('DTI_Risk_Bucket')['loan_paid_back'].mean()


from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

numeric_cols = [
    'annual_income',
    'debt_to_income_ratio',
    'credit_score',
    'loan_amount',
    'interest_rate',
    'loan_to_income'
]

X = df_train[numeric_cols].copy()

# Add constant
X = X.assign(constant=1)

vif_table = pd.DataFrame({
    "feature": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})

print(vif_table)


def compute_woe_iv(df, feature, target='loan_paid_back'):
    temp = df[[feature, target]].copy()
    
    # goods = 1, bads = 0
    grouped = temp.groupby(feature)[target].agg(['count', 'sum'])
    grouped.rename(columns={'sum': 'good'}, inplace=True)
    grouped['bad'] = grouped['count'] - grouped['good']

    # totals
    total_good = grouped['good'].sum()
    total_bad = grouped['bad'].sum()

    # percentages
    grouped['good_pct'] = grouped['good'] / total_good
    grouped['bad_pct'] = grouped['bad'] / total_bad

    # avoid divide-by-zero
    grouped['good_pct'] = grouped['good_pct'].replace(0, 1e-6)
    grouped['bad_pct'] = grouped['bad_pct'].replace(0, 1e-6)

    # WOE
    grouped['WOE'] = np.log(grouped['good_pct'] / grouped['bad_pct'])

    # IV
    grouped['IV'] = (grouped['good_pct'] - grouped['bad_pct']) * grouped['WOE']

    return grouped


categorical_features = [
    'gender',
    'marital_status',
    'education_level',
    'employment_status',
    'loan_purpose',
    'grade_subgrade',
    'credit_score_bucket',
    'DTI_Risk_Bucket'
]


iv_summary = []

woe_tables = {}  

for feat in categorical_features:
    woe_table = compute_woe_iv(df_train, feat, target='loan_paid_back')
    woe_tables[feat] = woe_table
    iv_value = woe_table['IV'].sum()
    iv_summary.append((feat, iv_value))


iv_df = pd.DataFrame(iv_summary, columns=['Feature', 'IV']).sort_values(by='IV', ascending=False)
iv_df


features_to_drop = [
    'gender',
    'marital_status',
    'education_level',
    'loan_purpose',
    'grade_subgrade'
]
df_train = df_train.drop(columns=features_to_drop)


pd.crosstab(df_train['employment_status'], df_train['loan_paid_back'], normalize='index')


df_train = df_train.drop(columns='employment_status')


df_train.columns, df_test.columns


df_test['grade'] = df_test['grade_subgrade'].str[0]              
df_test['sub']   = df_test['grade_subgrade'].str[1:].astype(int) 
df_test['loan_to_income'] = df_test['loan_amount'] / df_test['annual_income']
df_test['loan_to_income'] = df_test['loan_to_income'].replace([np.inf,-np.inf], np.nan)

bins = [300, 580, 670, 740, 800, 851]
labels = ["Poor","Fair","Good","Very Good","Exceptional"]

df_test['credit_score_bucket'] = pd.cut(
    df_test['credit_score'],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=False
)


bins_dti = [0.01, 0.05, 0.25, float('inf')]
labels_dti = ["Low", "Moderate", "High"]

df_test['DTI_Risk_Bucket'] = pd.cut(
    df_test['debt_to_income_ratio'],
    bins=bins_dti,
    labels=labels_dti,
    include_lowest=True,
    right=False
)


features_to_drop = [
    'gender',
    'marital_status',
    'education_level',
    'loan_purpose',
    'grade_subgrade',
]
df_test = df_test.drop(columns=features_to_drop)


df_test = df_test.drop(columns=['employment_status'])


df_train.columns, df_test.columns


numeric_features = [
    'annual_income',
    'debt_to_income_ratio',
    'credit_score',
    'loan_amount',
    'interest_rate',
    'loan_to_income',
    'sub'
]

categorical_features = [
    'grade',
    'credit_score_bucket',
    'DTI_Risk_Bucket'
]


df_train.sample(1)


iv_summary = []
woe_tables = {}

for feat in categorical_features:
    woe_table = compute_woe_iv(df_train, feat, target='loan_paid_back')
    woe_tables[feat] = woe_table
    
    iv_value = woe_table['IV'].sum()
    iv_summary.append((feat, iv_value))

iv_df = pd.DataFrame(iv_summary, columns=['Feature', 'IV'])
iv_df = iv_df.sort_values('IV', ascending=False)

iv_df


selected_features = [
    'annual_income',
    'debt_to_income_ratio',
    'credit_score',
    'loan_amount',
    'interest_rate',
    'loan_to_income',
    'sub',
    'grade',
    'credit_score_bucket',
    'DTI_Risk_Bucket'
]

TARGET = "loan_paid_back"

y = df_train[TARGET]

train_X = df_train[selected_features].copy()
test_X  = df_test[selected_features].copy()

categorical_features = ['sub', 'grade', 'credit_score_bucket', 'DTI_Risk_Bucket']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


oof_lgb = np.zeros(len(train_X))
oof_cat = np.zeros(len(train_X))
oof_xgb = np.zeros(len(train_X))

test_lgb = np.zeros(len(test_X))
test_cat = np.zeros(len(test_X))
test_xgb = np.zeros(len(test_X))


lgb_params = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "max_depth": -1,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary",
    "random_state": 42,
    "n_jobs": -1
}

cat_params = {
    "iterations": 800,
    "learning_rate": 0.03,
    "depth": 6,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "verbose": False,
    "random_state": 42
}

xgb_params = {
    "n_estimators": 900,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "objective": "binary:logistic",
    "tree_method": "hist",
    "random_state": 42
}


for fold, (tr_idx, val_idx) in enumerate(skf.split(train_X, y)):
    print(f"\n===== Fold {fold+1} / 5 =====")

    X_tr = train_X.iloc[tr_idx].copy()
    X_val = train_X.iloc[val_idx].copy()
    y_tr = y.iloc[tr_idx]
    y_val = y.iloc[val_idx]

    te = ce.TargetEncoder(cols=categorical_features, smoothing=0.25)
    te.fit(X_tr[categorical_features], y_tr)

    X_tr[categorical_features] = te.transform(X_tr[categorical_features])
    X_val[categorical_features] = te.transform(X_val[categorical_features])

    test_te = test_X.copy()
    test_te[categorical_features] = te.transform(test_te[categorical_features])

    lgbm = LGBMClassifier(**lgb_params)
    lgbm.fit(X_tr, y_tr)
    oof_lgb[val_idx] = lgbm.predict_proba(X_val)[:, 1]
    test_lgb += lgbm.predict_proba(test_te)[:, 1] / 5

    cat = CatBoostClassifier(**cat_params)
    cat.fit(X_tr, y_tr)
    oof_cat[val_idx] = cat.predict_proba(X_val)[:, 1]
    test_cat += cat.predict_proba(test_te)[:, 1] / 5


    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_tr, y_tr)
    oof_xgb[val_idx] = xgb.predict_proba(X_val)[:, 1]
    test_xgb += xgb.predict_proba(test_te)[:, 1] / 5


print("\nOOF AUC LGBM:", roc_auc_score(y, oof_lgb))
print("OOF AUC CAT :", roc_auc_score(y, oof_cat))
print("OOF AUC XGB :", roc_auc_score(y, oof_xgb))


oof_ensemble = (
    0.5 * oof_lgb +
    0.3 * oof_cat +
    0.2 * oof_xgb
)

test_ensemble = (
    0.5 * test_lgb +
    0.3 * test_cat +
    0.2 * test_xgb
)

print("\nENSEMBLE OOF AUC:", roc_auc_score(y, oof_ensemble))


def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.1, 0.9, 300)
    best_thr = 0.5
    best_auc = 0

    for thr in thresholds:
        curr_score = roc_auc_score(y_true, (y_prob > thr).astype(int))
        if curr_score > best_auc:
            best_auc = curr_score
            best_thr = thr
    return best_thr, best_auc

best_thr, best_thr_auc = find_best_threshold(y, oof_ensemble)

print("\nBest Threshold:", best_thr)
print("Best Threshold ROC-AUC:", best_thr_auc)


final_labels = (test_ensemble > best_thr).astype(int)


submission = pd.DataFrame({
    "id": df_test["id"],
    "loan_paid_back": final_labels
})

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("submission.csv saved!")


