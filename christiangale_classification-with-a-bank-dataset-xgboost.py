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

import warnings
warnings.filterwarnings('ignore')

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train_df.head()


train_df.info()


train_df.describe()


import matplotlib.pyplot as plt
import seaborn as sns


X = train_df.drop(columns=['id', 'y'], axis=1)
y = train_df['y']


def feature_engineer(df):
    df['balance_ishigh'] = (df['balance'] > np.median(df['balance'])).astype(int)
    df['duration_ishigh'] = (df['duration'] > np.median(df['duration'])).astype(int)
    return df


X = feature_engineer(X)


num_feat = X.select_dtypes(include='number')
cat_feat = X.select_dtypes(include='object')


for i in cat_feat.columns:
    print(f'{i}: {cat_feat[i].nunique()}')


plt.figure(figsize=(12, 8))

for i, col in enumerate([col for col in cat_feat.columns if X[col].nunique() <= 4]):
    plt.subplot(3, 3, i + 1)
    plt.pie(X[col].value_counts(), labels=X[col].value_counts().index, autopct='%1.1f%%',wedgeprops=dict(width=0.75), startangle=90, colors=sns.color_palette('pastel'))
    plt.title(col)
    plt.axis('equal')

plt.show()


plt.figure(figsize=(12, 10))

for i, col in enumerate(num_feat.columns):
    plt.subplot(5, 2, i + 1)
    sns.histplot(X[col], bins=30, kde=True, color='skyblue')
    plt.title(col)

plt.tight_layout()
plt.show()


sns.barplot(
    x=y.value_counts().index,
    y=y.value_counts().values,
    palette='pastel'
)
plt.show()


from sklearn.model_selection import train_test_split, KFold, cross_val_score


X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42, stratify=y)


oe_cols = cat_feat.drop(['education', 'month'], axis=1).columns
oe_education = 'education'
oe_month = 'month'


from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


preprocess = ColumnTransformer([
    ('le', OrdinalEncoder(), oe_cols),
    ('oe_education', OrdinalEncoder(categories=[['unknown', 'primary', 'secondary', 'tertiary']]), [oe_education]),
    ('oe_month', OrdinalEncoder(categories=[['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']]), [oe_month]),
    ('scaler', StandardScaler(), num_feat.columns),
])


from xgboost import XGBClassifier
pipeline = Pipeline([
    ('process', preprocess),
    ('model', XGBClassifier())
])


score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
print(score)


from functools import partial
import optuna
def objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'n_jobs': -1,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'tree_method':'gpu_hist',
        'device':'cuda',
    }
    oe_cols = cat_feat.drop(['education', 'month'], axis=1).columns
    oe_education = 'education'
    oe_month = 'month'

    preprocess = ColumnTransformer([
        ('le', OrdinalEncoder(), oe_cols),
        ('oe_education', OrdinalEncoder(categories=[['unknown', 'primary', 'secondary', 'tertiary']]), [oe_education]),
        ('oe_month', OrdinalEncoder(categories=[['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']]), [oe_month]),
        ('scaler', StandardScaler(), num_feat.columns),
    ])

    model = Pipeline([
        ('process', preprocess),
        ('model', XGBClassifier(
            **params,
            )
        )
    ])

    # Cross-validation score (you can use accuracy, f1, etc.)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    return score.mean()


objective_func = partial(objective, X=X_train, y=y_train)
study = optuna.create_study(direction='maximize')
study.optimize(objective_func, n_trials=20)


print("\nBest Parameters:", study.best_trial.params)
print("Best Cross-Validated Accuracy:", f"{study.best_value:.2%}")


train_pipe = Pipeline([
    ('process', preprocess),
    ('model', XGBClassifier(**study.best_params))
]).fit(X, y)


fe_test = feature_engineer(test_df)


result = train_pipe.predict_proba(fe_test)[:, 1]


sample = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv').set_index('id')


sample['y'] = result


submission = sample


submission.head()


submission.to_csv('submsission.csv')

