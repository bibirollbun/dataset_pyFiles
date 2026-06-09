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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

# Clean and normalize 'Sex'
train['Sex'] = train['Sex'].str.strip().str.lower()
test['Sex'] = test['Sex'].str.strip().str.lower()

# Feature engineering
def add_features(df):
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['HR_per_min'] = df['Heart_Rate'] / (df['Duration'].replace(0, np.nan))
    df['Temp_per_min'] = df['Body_Temp'] / (df['Duration'].replace(0, np.nan))
    df.fillna(0, inplace=True)
    return df

train = add_features(train)
test = add_features(test)

# Target
y = train['Calories']
X = train.drop(['Calories', 'id'], axis=1)
X_test = test.drop('id', axis=1)

# Label encoding
le = LabelEncoder()
X['Sex'] = le.fit_transform(X['Sex'])
X_test['Sex'] = le.transform(X_test['Sex'])

# Log target for RMSLE
y_log = np.log1p(y)

# K-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=31,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(100),
            log_evaluation(100)
        ]
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits

# Evaluate
rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_log), np.expm1(oof_preds)))
print(f'OOF RMSLE: {rmsle:.4f}')

# Submission
submission['Calories'] = np.expm1(test_preds).clip(0)  # clip negative values if any
submission.to_csv('submission.csv', index=False)



import matplotlib.pyplot as plt
train.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()


