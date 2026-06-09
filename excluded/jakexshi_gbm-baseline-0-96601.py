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

train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')



print(train.columns)


print(train.shape)
print(train.isnull().sum())
print(train['y'].value_counts(normalize=True))


cat_cols = train.select_dtypes('object').columns.tolist()
num_cols = train.select_dtypes(include=['int64', 'float64']).drop(['id', 'y'], axis=1).columns.tolist()


from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


from sklearn.model_selection import train_test_split

X = train.drop(['id', 'y'], axis=1)
y = train['y']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)



pip install --upgrade lightgbm


import lightgbm as lgb
from sklearn.metrics import roc_auc_score


model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.01)
model.fit(X_train, y_train, 
          eval_set=[(X_valid, y_valid)]
          # early_stopping_rounds=50,
          # verbose=100
         )

# Evaluate
val_preds = model.predict_proba(X_valid)[:, 1]
print("Validation ROC AUC:", roc_auc_score(y_valid, val_preds))



test_preds = model.predict_proba(test.drop(['id'], axis=1))[:, 1]
submission['y'] = test_preds
submission.to_csv('submission.csv', index=False)




