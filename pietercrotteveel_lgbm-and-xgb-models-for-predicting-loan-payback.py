from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import LGBMClassifier
import lightgbm as lgb
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV, KFold
import xgboost as xgb
from lightgbm import LGBMClassifier
import warnings
from scipy.stats import skew

warnings.filterwarnings('ignore')


test = pd.read_csv('/kaggle/input/data-sets/test.csv')
train = pd.read_csv('//kaggle/input/playground-series-s5e11/train.csv')
submission = pd.read_csv('/kaggle/input/data-sets/sample_submission.csv')
xgb_submission = pd.read_csv('/kaggle/input/data-sets/xgb_submission.csv')
lmg_submission = pd.read_csv('/kaggle/input/data-sets/lmg_submission.csv')



pd.set_option('display.max_columns', None)
train.head()


train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

print(f"df Train shape {train.shape}")
print(f"df Test shape {test.shape}")


print("Number of null value in Train DataFrames : ",train.isna().sum().sum())
print("Number of null value in Test DataFrames : ",test.isna().sum().sum())
print("Number of Duplicated Row in Train DataFrames : ", train.duplicated().sum())
print("Number of Duplicated Row in Test DataFrames : ", test.duplicated().sum())


num_cols = train.select_dtypes(exclude= 'object').columns
cat_cols = train.select_dtypes(include= 'object').columns

print('Numerical columns :', ', '.join(num_cols))
print("Number of numerical columns:" ,len(num_cols))
print("")
print('Categorical columns :', ', '.join(cat_cols))
print("Number of categorical columns:" ,len(cat_cols))


train[num_cols].describe().T


type(num_cols)


plt.figure(figsize=(8, 6))
correlation_matrix = train[num_cols ].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', linewidths=0.1, cmap="Blues")
plt.title('Correlation Heatmap')
plt.xlabel('Features')
plt.ylabel('Features')
plt.show()


num_cols=num_cols.drop('loan_paid_back')


plt.figure(figsize=(10, 8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(len(num_cols), 2, 2*i - 1)
    sns.histplot(train[col], kde=True, bins=40, color="#9dbbe2")
    plt.title(f'Distribution: {col}')

    plt.subplot(len(num_cols), 2, 2*i)
    sns.boxplot(x=train[col], color="#627cfc")
    plt.title(f'Boxplot: {col}')

plt.tight_layout()
plt.show()


skew_values = train[num_cols].apply(lambda s: skew(s.dropna(), bias=False))
sorted_skew_values = skew_values.sort_values(ascending=False)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(sorted_skew_values.index, sorted_skew_values.values)
plt.title('Skewness of Features')
plt.xlabel('Features')
plt.ylabel('Skewness')
plt.xticks(rotation=45)
plt.show()
print(sorted_skew_values)


right_skewed_cols = skew_values[skew_values > 1].index.tolist()

print("Highly skewed columns:", ', '.join(right_skewed_cols))

for col in right_skewed_cols:
    train[col] = np.log1p(train[col])
    test[col]  = np.log1p(test[col])

print("Log transformation applied to highly skewed columns.")




for col in num_cols:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    train[col] = train[col].clip(lower=lower_bound, upper=upper_bound)
    test[col] = test[col].clip(lower=lower_bound, upper=upper_bound)
print("Outliers removed using IQR method.")


plt.figure(figsize=(14, 8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)
    sns.kdeplot(train[col], label='Train', fill=True, alpha=0.5)
    sns.kdeplot(test[col], label='Test', fill=True, alpha=0.3)
    plt.title(f'Distribution comparison: {col}')
    plt.legend()
plt.tight_layout()
plt.show()


target_counts = train['loan_paid_back'].value_counts()
target_percent = train['loan_paid_back'].value_counts(normalize=True) * 100


plt.figure(figsize=(5,5))
bars = plt.bar(target_counts.index.astype(str),
               target_counts.values,
               color=["#4365b4","#6689ad"])

for bar in bars:
    height = bar.get_height()
    percent = (height / target_counts.sum()) * 100
    plt.text(bar.get_x() + bar.get_width()/2, height + 1000,  # adjust '1000' if scale differs
             f'{percent:.2f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

plt.title('Distribution of Loan Paid Back')
plt.xlabel('Loan Paid Back (1 = Yes, 0 = No)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


for col in cat_cols:
    print(f"\n Feature: {col}")

    freq = train[col].value_counts(dropna=False)
    repayment_rate = train.groupby(col)['loan_paid_back'].mean().sort_values(ascending=False)

    summary = pd.concat([freq, repayment_rate], axis=1)
    summary.columns = ['Count', 'Repayment_Rate']
    print("\nSummary:")
    print(summary)

    plt.figure(figsize=(8,4))
    
    sns.barplot(
        x=repayment_rate.index,
        y=repayment_rate.values,
        palette="Blues_d"
    )
    plt.title(f'Repayment Rate by {col}')
    plt.ylabel('Mean loan_paid_back (repayment rate)')
    plt.xlabel(col)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


train.head()


train['loan_to_income'] = train['loan_amount'] / (train['annual_income'] + 1)
test['loan_to_income'] = test['loan_amount'] / (test['annual_income'] + 1)

train['total_debt'] = train['debt_to_income_ratio'] * train['annual_income']
test['total_debt'] = test['debt_to_income_ratio'] * test['annual_income']

train['available_income'] = train['annual_income'] * (1 - train['debt_to_income_ratio'])
test['available_income'] = test['annual_income'] * (1 - test['debt_to_income_ratio'])

train['affordability'] = train['available_income'] / (train['loan_amount'] + 1)
test['affordability'] = test['available_income'] / (test['loan_amount'] + 1)

train['monthly_payment'] = train['loan_amount'] * (1 + train['interest_rate']/100) / 12
test['monthly_payment'] = test['loan_amount'] * (1 + test['interest_rate']/100) / 12

train['payment_to_income'] = train['monthly_payment'] / (train['annual_income']/12 + 1)
test['payment_to_income'] = test['monthly_payment'] / (test['annual_income']/12 + 1)

train['risk_score'] = (train['debt_to_income_ratio'] * 40 + (1 - train['credit_score']/850) * 30 + train['interest_rate'] * 2)
test['risk_score'] = (test['debt_to_income_ratio'] * 40 + (1 - test['credit_score']/850) * 30 + test['interest_rate'] * 2)


train['credit_interest'] = train['credit_score'] * train['interest_rate'] / 100
test['credit_interest'] = test['credit_score'] * test['interest_rate'] / 100

train['income_credit'] = np.log1p(train['annual_income']) * train['credit_score'] / 1000
test['income_credit'] = np.log1p(test['annual_income']) * test['credit_score'] / 1000

train['debt_loan'] = train['debt_to_income_ratio'] * np.log1p(train['loan_amount'])
test['debt_loan'] = test['debt_to_income_ratio'] * np.log1p(test['loan_amount'])

train['log_income'] = np.log1p(train['annual_income'])
test['log_income'] = np.log1p(test['annual_income'])

train['log_loan'] = np.log1p(train['loan_amount'])
test['log_loan'] = np.log1p(test['loan_amount'])

print("New features created successfully.")


train.head()


X = train.drop(columns='loan_paid_back',axis=1)
y = train['loan_paid_back']


cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test[col] = le.transform(test[col])


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import LGBMClassifier

params = dict(
    n_estimators=1320,
    learning_rate=0.05,
    num_leaves=93,
    max_depth=5,
    colsample_bytree=0.975,
    subsample=0.743,
    reg_alpha=2.95,
    reg_lambda=0.0022,
    random_state=42,
    n_jobs=-1,
    metric='auc',
    objective='binary',
    boosting_type='gbdt',
    verbosity=-1,
)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

roc_curves, fold_scores = [], []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    print(f"--- Fold {fold}/{skf.n_splits} ---")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred

    test_preds += model.predict_proba(test)[:, 1] / skf.n_splits

    auc = roc_auc_score(y_val, val_pred)
    fold_scores.append(auc)
    print(f"Fold {fold} AUC: {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_val, val_pred)
    roc_curves.append((fpr, tpr, auc))

overall_auc = roc_auc_score(y, oof_preds)
print("Fold AUCs:", [round(s, 4) for s in fold_scores])
print(f"Overall OOF AUC: {overall_auc:.5f}")


final_model = LGBMClassifier(**params)
final_model.fit(X, y)


xgb_params = dict(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    max_depth=6,
    learning_rate=0.0669438421783529,
    n_estimators=732,
    min_child_weight=8.368496274182363,
    subsample=0.8638990746572127,
    colsample_bytree=0.9262609574627299,
    gamma=1.9880100566380507,
    reg_alpha=0.010470012214699875,
    reg_lambda=0.010061409517576274,
    max_bin=504,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

xgb_model = xgb.XGBClassifier(**xgb_params)

oof_preds = np.zeros(len(X))
xgb_test_preds = np.zeros(len(test))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
roc_curves, fold_scores = [], []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    print(f"--- Fold {fold}/{skf.n_splits} ---")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    

    xgb_val_pred = xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = xgb_val_pred

    xgb_test_preds += xgb_model.predict_proba(test)[:, 1] / skf.n_splits

    fpr, tpr, _ = roc_curve(y_val, xgb_val_pred)
    roc_curves.append((fpr, tpr, auc))

    auc = roc_auc_score(y_val, xgb_val_pred)
    fold_scores.append(auc)
    print(f"Fold {fold} AUC: {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_val, xgb_val_pred)
    roc_curves.append((fpr, tpr, auc))


simple_avg_score = np.mean(fold_scores)
overall_auc = roc_auc_score(y, oof_preds)

print("Fold AUCs:", [round(s, 4) for s in fold_scores])
print(f"Overall OOF AUC: {overall_auc:.5f}")




lmg_submission['loan_paid_back'] = test_preds
lmg_submission.to_csv('lmg_submission.csv', index=False)
lmg_submission.head()


xgb_submission['loan_paid_back'] = xgb_test_preds
xgb_submission.to_csv('xgb_submission.csv', index=False)
xgb_submission.head()

