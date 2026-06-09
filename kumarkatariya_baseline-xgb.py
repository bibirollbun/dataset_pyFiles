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


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')


test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


test_ids = test['id']


test = test.drop('id',axis=1)


train.info()


train_ids = train['id']



train = train.drop('id',axis=1)


target = train['diagnosed_diabetes']


train = train.drop('diagnosed_diabetes',axis=1)


train.head()


test.head()


num_cols = train.select_dtypes(include = ['int','float']).columns.to_list()


num_cols


cat_cols = train.select_dtypes(include= 'object').columns.to_list()


cat_cols


X = train.copy()
y = target.copy()


for col in cat_cols: 
    combined = pd.concat([X[col],test[col]],axis=0) 
    all_cat = pd.Categorical(combined).categories
    cat_type = pd.CategoricalDtype(categories=all_cat)
    X[col] = X[col].astype(cat_type)
    test[col] = test[col].astype(cat_type)


n_splits = 5
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=n_splits,random_state=42,shuffle=True)



from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


params = {
    'max_depth': 5,
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'early_stopping_rounds': 100,
    'random_state': 42, 
    'enable_categorical': True
}


oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

for train_idx,val_idx in skf.split(X,y):
    X_train,X_val = X.iloc[train_idx],X.iloc[val_idx]
    y_train,y_val = y.iloc[train_idx],y.iloc[val_idx] 
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_metric='auc',
              eval_set=[(X_train, y_train),(X_val, y_val)],
              verbose=1000)
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:,1]
    test_preds+= model.predict_proba(test)[:,1]/n_splits
print('auc',roc_auc_score(y,oof_preds))    


sample = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


sample.head()


submission = pd.DataFrame({'id':test_ids,
             'diagnosed_diabetes':test_preds})


submission.to_csv('submission.csv',index=False)




