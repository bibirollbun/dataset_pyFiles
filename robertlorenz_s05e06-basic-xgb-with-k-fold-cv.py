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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv", index_col="id")


cat_cols = [col for col in train.select_dtypes(include=['object']).columns if col != "Fertilizer Name"]

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

label_encoder = LabelEncoder()
train['Fertilizer Name'] = label_encoder.fit_transform(train['Fertilizer Name'])


K = 10

X = train.drop('Fertilizer Name', axis=1)
y = train['Fertilizer Name']

kcv = KFold(n_splits=K, shuffle=True, random_state=42)

for i, (train_idx, val_idx) in enumerate(kcv.split(X, y)):
    print(f"\n{'-'*5}Fold {i+1}/{K}{'-'*5}")

    X_train_ = X.iloc[train_idx].reset_index(drop=True)
    y_train_ = y.iloc[train_idx].reset_index(drop=True)
    X_val_ = X.iloc[val_idx].reset_index(drop=True)
    y_val_ = y.iloc[val_idx].reset_index(drop=True)

    xgboost_params = {
        'alpha': 4.93, 
        'colsample_bytree': 0.6, 
        'early_stopping_rounds': 328, 
        'eta': 0.0127, 
        'gamma': 0.232, 
        'max_delta_step': 5.62, 
        'max_depth': 22, 
        'min_child_weight': 6.916, 
        'n_estimators': 7206, 
        'reg_lambda': 1.28, 
        'subsample': 0.94,
        'device': 'cuda',
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'n_jobs': -1,}

    model = xgb.XGBClassifier(**xgboost_params)

    model.fit(X_train_, y_train_, eval_set=[(X_val_, y_val_)], verbose=500)

    y_pred_proba = model.predict_proba(X_val_)
    # MAP@3 evaluation
    map_score = 0.0
    for j in range(len(val_idx)):
        # Get top 3 predictions for this sample
        top_3_preds = np.argsort(y_pred_proba[j])[::-1][:3]
        
        correct = 0
        precision = 0.0
        
        for k, pred in enumerate(top_3_preds):
            if pred == y_val_[j]:
                correct += 1
                precision += correct / (k + 1)
        
        # Average precision for this sample
        if correct > 0:
            map_score += precision / min(1, correct)

    print(f'Fold {i+1}: Map@3 score: {map_score / len(val_idx)}')


X_test = test
test_ids = test.index
test_preds = model.predict_proba(X_test)
top3_idx = np.argsort(-test_preds, axis=1)[:, :3]
labels = label_encoder.inverse_transform(top3_idx.ravel())
pred_names = labels.reshape(top3_idx.shape)

submission_format = [' '.join(row) for row in pred_names]

submission = pd.DataFrame({'id': test_ids, 'Fertilizer Name': submission_format})
submission.to_csv('submission.csv', index=False)
submission.head()

