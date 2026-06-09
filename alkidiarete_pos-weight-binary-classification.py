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
from sklearn.preprocessing import LabelEncoder
from numpy import unique
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

test_id = test['id']
train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)


train.head()


numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'] 
cat_cols = ['job','marital','education','default','housing','loan','contact','month','poutcome']

def preprocess(df):
    
    for col in numerical_cols:
        lower_bound = df[col].quantile(0.005)
        upper_bound = df[col].quantile(0.995)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['has_previous'] = (df['previous'] > 0).astype(int)
    df['duration_long'] = (df['duration'] > 300).astype(int)
    df['campaign_multiple'] = (df['campaign'] > 2).astype(int)
    df['sqrt_age'] = np.sqrt(df['age'])
    
    df['duration_log']=np.log(df['duration'])
    df['campaign_log']=np.log(df['campaign'])
    df['pdays_log']=np.log(df['pdays']+2)
    df['previous_log']=np.log(df['previous']+1)
    
    for feature in cat_cols:
        df[feature] = df[feature].astype("category")
    
    return df

train=preprocess(train)
test=preprocess(test)


label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]], axis=0))
    
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])


X = train.drop('y', axis=1)
y = train['y']
X_test = test.copy()


classes = unique(y)
total = len(y)
for c in classes:
	n_examples = len(y[y==c])
	percent = n_examples / total * 100
	print('> Class=%d : %d/%d (%.1f%%)' % (c, n_examples, total, percent))


scale_pos_weight = (y == 0).sum() / (y == 1).sum()

best_params = {
    'max_depth': 7,
    'learning_rate': 0.12439245978422549,
    'n_estimators': 634,
    'subsample': 0.8420253692354978,
    'colsample_bytree': 0.5913728441717878,
    'gamma': 2.8041665914251728,
    'min_child_weight': 6,
    'reg_alpha': 2.733065659441194,
    'reg_lambda': 2.853934820707197,
    'scale_pos_weight': scale_pos_weight,
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'random_state': 42,
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor'
}

model = XGBClassifier(**best_params)


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y, scoring="roc_auc", cv=cv, n_jobs=1)

print(f"Mean ROC AUC: {np.mean(scores):.5f}")


model = XGBClassifier(**best_params)
model.fit(X, y)

y_pred = model.predict_proba(test)[:, 1]


submission = pd.DataFrame({
    "id": test_id, 
    "y": y_pred
})

submission.to_csv("submission.csv", index=False)

submission.head()


