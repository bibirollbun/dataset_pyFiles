# --- Import Libraries
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import gc, time

import warnings
warnings.filterwarnings('ignore')


# --- Configuration
SEED = 42
FOLDS = 5

sns.set_style("whitegrid")


# --- Load dataset
train_df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test_df  = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
print("train", train_df.shape, "test", test_df.shape)


print("======== TRAIN ========")
print(train_df.isnull().sum(), "\n")
print("======== TEST ========")
print(test_df.isnull().sum())


print("======== TRAIN ========")
print(train_df.dtypes, "\n")
print("======== TEST ========")
print(test_df.dtypes)


plt.figure(figsize=(10, 5))
bar = sns.countplot(data=train_df, x='loan_paid_back', palette='Set1')

counts = train_df['loan_paid_back'].value_counts().sort_index()
ratios = counts / counts.sum() * 100

for p, (cls, ratio) in zip(bar.patches, ratios.items()):
    height = p.get_height()
    bar.annotate(
        f"{int(height):,}\n({ratio:.1f}%)",
        (p.get_x() + p.get_width() / 2., height),
        ha='center', va='bottom',
        fontsize=12, fontweight='bold',
        color='black', xytext=(0, 4),
        textcoords='offset points'
    )

plt.xlabel('loan_paid_back')
plt.ylabel('Count')
plt.tight_layout()
plt.show()



cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']

plt.figure(figsize=(15, 15))
for i, col in enumerate(cols):
    plt.subplot(3, 2, i+1)

    bar = sns.countplot(x=col, hue='loan_paid_back', data=train_df, palette='Set1')
    for i in bar.containers:
        bar.bar_label(i)

    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()       


grade_summary = (
    train_df.groupby(['grade_subgrade', 'loan_paid_back'])
    .size()
    .reset_index(name='count')
)

grade_summary['ratio'] = (
    grade_summary['count'] / 
    grade_summary.groupby('grade_subgrade')['count'].transform('sum')
) * 100

plt.figure(figsize=(14, 6))
sns.barplot(
    data=grade_summary,
    x='grade_subgrade',
    y='ratio',
    hue='loan_paid_back',
    palette='Set1'
)

for container in plt.gca().containers:
    plt.bar_label(container, fmt='%.1f%%', label_type='edge', fontsize=9)

plt.title('Repayment Ratio by Grade_Subgrade', fontsize=14)
plt.xlabel('Grade_Subgrade')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.legend(
    title='Loan Paid Back',
    title_fontsize=12,
    fontsize=11,
    loc='upper right'
)
plt.tight_layout()
plt.show()


cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']

train_df[cols].describe()


plt.figure(figsize=(10, 10))
for i, col in enumerate(cols):
    plt.subplot(3, 2, i+1)

    sns.histplot(train_df[col], kde=True, palette='Set1')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()   


plt.figure(figsize=(12, 12))

for i, col in enumerate(cols):
    plt.subplot(3, 2, i+1)
    sns.boxplot(data=train_df, x='loan_paid_back', y=col, palette='Set2')
    plt.xlabel(f'{col} - Loan Paid Back (0=No, 1=Yes)')
    plt.ylabel(f'{col}')

plt.tight_layout()
plt.show()


#--- debt_to_income_ratio
## binning
bins = np.arange(0, 0.65, 0.05)
binning = pd.cut(train_df['debt_to_income_ratio'], bins=bins)
score_summary = train_df.groupby(binning)['loan_paid_back'].mean().reset_index()

plt.figure(figsize=(6, 4))
sns.barplot(data=score_summary, x='debt_to_income_ratio', y='loan_paid_back', color='green')
plt.xlabel('Debt to Income Ratio Range')
plt.ylabel('Repayment Rate')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.show()

print("\n\n")
#--- credit_score
## binning
bins = np.arange(350, 900, 50)
binning = pd.cut(train_df['credit_score'], bins=bins)
score_summary = train_df.groupby(binning)['loan_paid_back'].mean().reset_index()

plt.figure(figsize=(6, 4))
sns.barplot(data=score_summary, x='credit_score', y='loan_paid_back', color='red')
plt.xlabel('Credit Score Range')
plt.ylabel('Repayment Rate')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.show()


#--- copy
train = train_df.copy()
test = test_df.copy()


#--- One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder

cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']
encoder = OneHotEncoder(sparse=False, drop=None, handle_unknown='ignore')

# train data
encoded_train = encoder.fit_transform(train[cols])
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(cols))
train = pd.concat([train.drop(columns=cols), encoded_train_df], axis=1)

# test data
encoded_test = encoder.transform(test[cols])
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(cols))
test = pd.concat([test.drop(columns=cols), encoded_test_df], axis=1)


#--- Ordinal Encoding -> grade_subgrade
# mapping the grades into ordered integers
order = ["A1", "A2", "A3", "A4", "A5", 
         "B1", "B2", "B3", "B4", "B5",
         "C1", "C2", "C3", "C4", "C5",
         "D1", "D2", "D3", "D4", "D5",
         "E1", "E2", "E3", "E4", "E5",
         "F1", "F2", "F3", "F4", "F5"
        ]

# Mapping
mapping = {v: i for i, v in enumerate(order)}
train['grade_subgrade'] = train['grade_subgrade'].map(mapping)
test['grade_subgrade'] = test['grade_subgrade'].map(mapping)


### I will add new features as needed...


X = train.drop(columns=['id', 'loan_paid_back'])
y = train['loan_paid_back']


import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

roc_scores = []
models = []
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc", 
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"========= \nFold {fold+1}/{FOLDS}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=100,
        verbose_eval=0
    )

    y_pred_val = model.predict(dval, iteration_range=(0, model.best_iteration))
    
    # Calculate the score
    score = roc_auc_score(y_val, y_pred_val)
    print(f'Fold: {fold+1} AUC score: {np.mean(score):.5f}') 

    roc_scores.append(score)
    models.append(model)


print(f'\nAverage AUC Score : {np.mean(roc_scores):.5f}, +-: {np.std(roc_scores):.5f}')


X_test = test.drop(columns=["id"])
submit_score = []

dtest = xgb.DMatrix(X_test)
for fold_, model in enumerate(models):
    # predict test data
    pred_ = model.predict(dtest, iteration_range=(0, model.best_iteration))
    submit_score.append(pred_)

# predict test data
pred = np.mean(submit_score, axis=0)


submission = pd.DataFrame({
    'id': test_df['id'],
    'loan_paid_back': pred
})

# Save
submission.to_csv('submission.csv', index=False)


submission

