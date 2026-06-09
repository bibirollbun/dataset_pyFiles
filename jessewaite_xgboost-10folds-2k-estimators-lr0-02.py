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
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


training_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
testing_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


for df in [training_df, testing_df]:
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
    df['duration_squared'] = df['duration'] ** 2
    df['duration_log'] = np.log1p(df['duration'])
    df['duration_sqrt'] = np.sqrt(df['duration'])


testing_df.drop(columns=['id'], inplace=True)
y = training_df['y']


training_df.head()


training_df.info()


objects = training_df.select_dtypes('object').columns

for obj in objects:
    le = LabelEncoder()
    le.fit(pd.concat([training_df[obj],testing_df[obj]], axis=0).astype(str))
    training_df[obj] = le.transform(training_df[obj].astype(str))
    testing_df[obj] = le.transform(testing_df[obj].astype(str))


training_df.info()


X = training_df.drop(columns=['id','y'])


params = {'n_estimators':2000,
     'learning_rate':0.02,
     'max_depth':8,
     'min_child_weight':2,
     'colsample_bytree':0.9,
     'subsample':0.9,
     'gamma':1.5,
     'reg_alpha':0.2,
     'reg_lambda':2,
     'scale_pos_weight':1,
     'tree_method':'hist',
     'max_bin':256,
     'grow_policy':'lossguide',
     'random_state':42,
     'use_label_encoder':False,
     'eval_metric':'auc',
     'objective':'binary:logistic',
     'early_stopping_rounds':30}


test_pred = np.zeros(len(testing_df))
folds = 10
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, 
              eval_set=[(X_test,y_test)],
              verbose=False)

    # changing from predict to predict_proba boosted the score by almost 10%
    test_pred += model.predict_proba(testing_df)[:,1]/folds

    y_pred = model.predict_proba(X_test)[:,1]
    print(f'Fold {i+1} AUC score: {roc_auc_score(y_test, y_pred):.4f}')


submission_ID = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
data = {'id': submission_ID['id'],
        'y':test_pred}

df = pd.DataFrame(data)
df.to_csv('submission.csv', index=False)
df[:5]

