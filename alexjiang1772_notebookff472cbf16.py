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
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# 读取数据
train_df = pd.read_csv("/kaggle/input/playground-series-s3e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s3e3/test.csv")


train_df


train_df.drop(columns=['id'], inplace=True)
test_id = test_df['id']  
test_df.drop(columns=['id'], inplace=True)

train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)


common_cols = train_df.columns.intersection(test_df.columns).tolist()
common_cols.append('Attrition')
train_df = train_df[common_cols]
test_df = test_df[common_cols[:-1]]

X = train_df.drop(columns=['Attrition'])
y = train_df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.5, random_state=42)  
X_train, y_train = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
test_scaled = scaler.transform(test_df)



train_data = lgb.Dataset(X_train_scaled, label=y_train)
valid_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)

params = {
    'objective': 'binary',  
    'metric': 'auc',  
    'boosting_type': 'gbdt', 
    'learning_rate': 0.05,
    'num_leaves': 31,  
    'max_depth': -1,  
    'min_child_samples': 20,  
    'subsample': 0.8,  
    'colsample_bytree': 0.8,  
    'reg_alpha': 0.1, 
    'reg_lambda': 0.1, 
    'random_state': 42
}

model = lgb.train(params, train_data, valid_sets=[train_data, valid_data])

y_pred_proba = model.predict(X_test_scaled)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("LightGBM ROC AUC Score:", roc_auc)

test_predictions_proba = model.predict(test_scaled)



submission_df = pd.DataFrame({'EmployeeNumber': test_id, 'Attrition': test_predictions_proba})
submission_df.to_csv("submission_lgbm.csv", index=False)

