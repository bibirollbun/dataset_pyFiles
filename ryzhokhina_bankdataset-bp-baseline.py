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


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv', index_col = 0)


train.info()


train.columns


train.describe()


train.head()


print(f"Missing values per column:\n{train.isna().sum()}")


print("\nTarget distribution:")
print(train['y'].value_counts(normalize=True))

sns.countplot(data=train, x='y')
plt.title("Target Variable Distribution")
plt.show()


num_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols.remove('y') if 'y' in num_cols else None  # remove target if numeric

# Summary statistics
train[num_cols].describe()


inf_counts = train.isin([np.inf, -np.inf]).sum()
inf_counts


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Histograms
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(train[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()


# Correlation heatmap
plt.figure(figsize=(10,6))
#corr = train[num_cols + ['y'].copy()] if 'y' in train.columns else train[num_cols]
sns.heatmap(train[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap - Numerical Features")
plt.show()


cat_cols = train.select_dtypes(['object']).columns.to_list()
cat_cols

for col in cat_cols:
    plt.figure(figsize = (8,4))
    order = train[col].value_counts().index
    ax = sns.countplot(data = train, x = col, order = order)
    ax.bar_label(ax.containers[0])
    plt.title(f"{col} distribution")
    plt.xticks(rotation = 45)
    plt.show()


train.head()


for col in num_cols:
    plt.figure(figsize = (12,8))
    sns.boxplot(data = train, x = 'y', y = col)
    plt.title(f'{col} vs target')
    plt.show()


# Categorical vs Target
for col in cat_cols:
    plt.figure(figsize=(8,4))
    prop_df = (train.groupby(col)['y']
                 .value_counts(normalize=True)
                 .rename('proportion')
                 .reset_index())
    sns.barplot(data=prop_df, x=col, y='proportion', hue='y')
    plt.title(f"{col} vs Target Proportion")
    plt.xticks(rotation=45)
    plt.show()


test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col = 'id')
print(test.shape)
test.head()


train.head()


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


drop_cols = ['contact', 'day', 'month']  # not predictive
#train.drop(columns=drop_cols, inplace=True)
#test.drop(columns=drop_cols, inplace=True)

# Encode categorical variables
cat_cols = train.select_dtypes('object').columns.to_list()
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# Features / target
X = train.drop('y', axis=1)
y = train['y']
X_test = test.copy()

# Stratified KFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

early_stopping_callback = lgb.early_stopping(stopping_rounds=10, verbose=True)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[early_stopping_callback]
    )
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / kf.n_splits

# Local validation AUC
auc_score = roc_auc_score(y, oof_preds)
print(f"CV ROC AUC: {auc_score:.4f}")


test_preds


submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


submission.head()


submission['y'] = test_preds


submission.head()


submission.to_csv("submission.csv", index=False)

