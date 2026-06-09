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


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import lightgbm as lgb
from datetime import datetime

# Load data
train = pd.read_csv("/kaggle/input/predicting-the-price-of-diamond/train.csv")
test = pd.read_csv("/kaggle/input/predicting-the-price-of-diamond/test.csv")
submit = pd.read_csv('/kaggle/input/predicting-the-price-of-diamond/submission.csv')

# Label encode categorical features
def encode(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col].fillna('N'))
    return df

data = encode(pd.concat([train, test], axis=0))
train_data = data.iloc[:len(train)]
test_data = data.iloc[len(train):]

y = train_data['price']
X = train_data.drop('price', axis=1)
X_test = test_data.drop('price', axis=1)

# Optimized parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.08925822012051833,
    'num_leaves': 233,
    'max_depth': 3,
    'min_child_samples': 52,
    'feature_fraction': 0.734916416713794,
    'bagging_fraction': 0.6725804874078269,
    'bagging_freq': 3,
    'lambda_l1': 0.0011416222074710496,
    'lambda_l2': 10.036185714981752,
    'colsample_bytree': 0.37754337369193036,
    'subsample': 0.6488697232880127,
    'subsample_freq': 3,
    'n_estimators': 4000,
    'random_state': 42,
    'verbosity': -1
}

print("Training with 10-fold CV...")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
oof = np.zeros(len(X))
preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)]
    )
    
    oof[val_idx] = model.predict(X_val)
    preds += model.predict(X_test) / 10
    
    print(f"Fold {fold+1}: R2 = {r2_score(y_val, oof[val_idx]):.5f}")

r2 = r2_score(y, oof)
print(f"\nFinal OOF R2: {r2:.5f}")

# Save submission
submit['price'] = preds
submit.to_csv('submission.csv', index=False)
print(f"Saved: submission.csv (OOF R2: {r2:.5f})")

