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


train_data = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col="id")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
train_data.head(10)


train_data.isnull().sum()


train_data.y.value_counts()


train_data.nunique()


def data_process(data):
    data['balance_sin'] = np.sin(2*np.pi * data['balance'] / 800)
    data['balance_cos'] = np.cos(2*np.pi * data['balance'] / 800)
    
    data['is_new_customer'] = (data['pdays'] == -1).astype(int)
    data['debt'] = data['housing']  + data['loan']
    
    data['duration_sin'] = np.sin(2*np.pi * data['duration'] / 800)
    data['duration_cos'] = np.cos(2*np.pi * data['duration'] / 800)
    
    return data
    
train_data = data_process(train_data)
test_data = data_process(test_data)
train_data.head()


cat_col = [cat for cat in train_data.columns if train_data[cat].dtype=="object"]
for col in cat_col:
    print(f"{col}: {train_data[col].nunique()} unique values")
    print(f"{sorted(train_data[col].unique())}\n")


from sklearn.metrics import roc_auc_score, f1_score , accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

X = train_data.copy()
y = train_data.y
X.drop(columns="y", axis=1, inplace=True)
X[cat_col] = X[cat_col].astype('category')

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, shuffle=True, stratify=y)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=4, shuffle=True, stratify=y_train)


import xgboost as xgb

model = xgb.XGBClassifier(
        enable_categorical=True,
        tree_method='hist',
        device='cuda',
        n_estimators=100000,
        learning_rate=0.01,
        max_depth=8,
        subsample=0.6,
        colsample_bytree=0.5,
        # scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='auc',
        early_stopping_rounds=300,
        random_state=42
    )


# model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

# y_pred = model.predict(X_val)
# y_prob = model.predict_proba(X_val)[:, 1]
# f1 = f1_score(y_val, y_pred)
# precision = precision_score(y_val, y_pred)
# recall = recall_score(y_val, y_pred)
# auc_score = roc_auc_score(y_val, y_prob)
# print(f"模型的 F1 分数是: {f1:.4f}")
# print(f"模型的 ROC AUC 分数是: {auc_score:.4f}")
# print(f"模型的 Precision 分数是: {precision:.4f}")
# print(f"模型的 Recall 分数是: {recall:.4f}")


# y_pred = model.predict(X_test)
# y_prob = model.predict_proba(X_test)[:, 1]
# f1 = f1_score(y_test, y_pred)
# auc_score = roc_auc_score(y_test, y_prob)
# print(f"模型的测试集 F1 分数是: {f1:.4f}")
# print(f"模型的测试集 ROC AUC 分数是: {auc_score:.4f}")


from sklearn.model_selection import StratifiedKFold

test = test_data.copy()
test.drop(columns="id", axis=1, inplace=True)
test[cat_col] = test[cat_col].astype('category')

n_splits = 10
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
probs = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f'===============Fold:{fold+1}===============')
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        verbose=100
    )
    
    probs += model.predict_proba(test)[:, 1] / n_splits


submission = pd.DataFrame({"id": test_data["id"], "y": probs})
submission.to_csv("submission.csv", index=False)
submission.head()

