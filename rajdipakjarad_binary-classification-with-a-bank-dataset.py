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


# Imports & Setup
import pandas as pd
import numpy as np
import os
import random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
pd.set_option('display.max_columns', 100)


TRAIN_PATH = '/kaggle/input/playground-series-s5e8/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e8/test.csv'
SUB_PATH = '/kaggle/input/playground-series-s5e8/sample_submission.csv'

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SUB_PATH)

train.shape, test.shape


display(train.head())
display(test.head())


print(train.info())
print(train.isnull().sum())
print(test.isnull().sum())


sns.countplot(x='y', data=train)
plt.title('Target Distribution (0 = No, 1 = Yes)')
plt.show()

print(train['y'].value_counts(normalize=True))


cat_features = train.select_dtypes(include='object').columns.tolist()
cat_features = [c for c in cat_features if c != 'y' and c != 'id']
print(cat_features)

# Numerical columns
num_features = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_features = [c for c in num_features if c not in ['y', 'id']]
print(num_features)


train[num_features].describe().T


for col in cat_features:
    print(f'{col}: {train[col].nunique()} unique values')
    print(train[col].value_counts(), "\n")


train['is_train'] = True
test['is_train'] = False
data_all = pd.concat([train, test], axis=0, ignore_index=True)
print(data_all.shape)


for col in cat_features:
    le = LabelEncoder()
    data_all[col] = le.fit_transform(data_all[col].astype(str))


train = data_all[data_all['is_train']].copy()
test = data_all[~data_all['is_train']].copy()
train.drop(['is_train'], axis=1, inplace=True)
test.drop(['is_train', 'y'], axis=1, errors='ignore', inplace=True)

# Reindex
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)


plt.figure(figsize=(12,4))
sns.histplot(train['balance'], bins=50)
plt.title("Balance Distribution")
plt.show()


plt.figure(figsize=(6,4))
sns.countplot(x='y', data=train)
plt.title('Target Distribution (Subscribed or Not)')
plt.xlabel('Subscribed (y)')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(10,6))
sns.histplot(data=train, x="age", hue="y", bins=30, stat="density", common_norm=False, kde=True)
plt.title('Age Distribution by Subscription Outcome')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend(labels=["Not Subscribed", "Subscribed"])
plt.show()


train['duration_per_campaign'] = train['duration'] / (train['campaign']+1)
test['duration_per_campaign'] = test['duration'] / (test['campaign']+1)


features = [c for c in train.columns if c not in ['id', 'y']]
X = train[features]
y = train['y']
X_test = test[features]


NFOLDS = 5
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)


def train_lgb(X, y, X_test, skf, cat_cols=None):
    oof = np.zeros(X.shape[0])
    preds = np.zeros(X_test.shape[0])
    scores = []
    models = []
    
    lgb_params = {
        'objective': 'binary',
        'learning_rate': 0.01,
        'n_estimators': 5000,
        'reg_alpha': 1,
        'reg_lambda': 1,
        'random_state': SEED,
        'n_jobs': -1,
        'metric': 'auc',
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_samples': 20,
        'device': 'cpu',
        'verbose': -1
    }
    
    from lightgbm import early_stopping

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**lgb_params)

       

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[early_stopping(stopping_rounds=100)]
        )

        oof[val_idx] = model.predict_proba(X_val)[:,1]
        preds += model.predict_proba(X_test)[:,1] / NFOLDS
        score = roc_auc_score(y_val, oof[val_idx])
        scores.append(score)
        print(f'Fold {fold+1} AUC: {score:.5f}')
        models.append(model)
    print(f'Mean CV AUC: {np.mean(scores):.5f} +- {np.std(scores):.5f}')
    return oof, preds, models


oof_lgb, test_lgb, lgb_models = train_lgb(X, y, X_test, skf)


final_predictions = test_lgb


plt.figure(figsize=(10,6))
sns.countplot(y='job', data=train, order=train['job'].value_counts().index, palette="viridis")
plt.title('Number of Clients by Job Category')
plt.xlabel('Count')
plt.ylabel('Job')
plt.show()


numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'y' in numeric_cols: numeric_cols.remove('y')
if 'id' in numeric_cols: numeric_cols.remove('id')
plt.figure(figsize=(10,8))
sns.heatmap(train[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title('Correlation Heatmap of Numeric Features')
plt.show()


submission = sample_submission.copy()
submission['y'] = final_predictions
submission.to_csv('submission.csv', index=False)
submission.head()




