import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import tensorflow as tf
from tensorflow.keras import models, layers, callbacks
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


# top rows
train_df.head()


# drop id's

train_df = train_df.drop('id', axis=1)

tid = test_df['id']
test_df = test_df.drop('id', axis=1)


# shape

print(f'rows: {train_df.shape[0]}\ncolumns: {train_df.shape[1]}')


# information
train_df.info()


# statistical information
train_df.describe()


# check duplicate

print(train_df.duplicated().sum())


train_df.head(2)


# univariate analysis | outliers | numeric data

ncols = train_df.select_dtypes(include='number').columns.to_list()
colors = sns.color_palette('husl', len(ncols))

plt.figure(figsize=(10, 8))
for i, col in enumerate(ncols):
    plt.subplot(3, 3, i+1)
    sns.violinplot(data=train_df, x=col, color=colors[i], saturation=1)
    plt.title(f'{col}')
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.8)
plt.show()


# univariate analysis | frequency distribution | numeric data

plt.figure(figsize=(10, 8))
for i, col in enumerate(ncols):
    plt.subplot(3, 3, i+1)
    sns.kdeplot(data=train_df, x=col, fill=True, color=colors[i])
    plt.title(f'{col} distribution')
    plt.xlabel(f'{col}')
    plt.ylabel('frequency')
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.subplots_adjust(wspace=0.5, hspace=0.8)
plt.show()


# skewness

train_df[ncols].skew()


# target variable analysis

ratio = train_df['y'].value_counts()
print(ratio)

neg = ratio[0]
pos = ratio[1]

plt.figure(figsize=(5, 4))
sns.countplot(x=train_df['y'], palette='rocket')
plt.show()


train_df['education'].value_counts()


# encoding

le = LabelEncoder()
cols = ['job', 'marital', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for col in cols:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])


ord = OrdinalEncoder(categories=[['unknown', 'primary', 'secondary', 'tertiary']])
train_df[['education']] = ord.fit_transform(train_df[['education']])
test_df[['education']] = ord.transform(test_df[['education']])


train_df.head(2)


# split

X = train_df.drop('y', axis=1)
y = train_df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# model init

models = {
    'xgboost': XGBClassifier(random_state=42, eval_metric='auc', use_label_encoder=False),
    'lightgbm': LGBMClassifier(class_weight='balanced', random_state=42)
}


# cross validation

cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
scoring = ['accuracy', 'roc_auc', 'f1']

best_model = None
best_score = 0
best_metric = 'roc_auc' 

for name, model in models.items():
    if hasattr(model, 'set_params'):
        model.set_params(verbosity=0)
    if hasattr(model, 'verbose'):
        model.verbose = -1
        
    score = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    
    avg_scores = {m: score[f'test_{m}'].mean() for m in scoring}
    print(f"{name} : " + " | ".join([f"{m}: {avg_scores[m]:.5f}" for m in scoring]))
    
    if avg_scores[best_metric] > best_score:
        best_score = avg_scores[best_metric]
        best_model = name

print(f"\nBest model based on {best_metric}: {best_model} ({best_metric} = {best_score:.5f})")


# best model | training

final_model = XGBClassifier(random_state=42, eval_metric='auc', use_label_encoder=False)
final_model.fit(X, y)


# prediction

y_proba = final_model.predict_proba(test_df)[:, 1] 
y_proba


# submission file

submission = pd.DataFrame({
    'id': tid,
    'y': y_proba
})

submission.head()


# save submission to csv

submission.to_csv('submission.csv', index=False)

