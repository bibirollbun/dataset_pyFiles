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

tr = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv');
te = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv');


tr.head()


y=tr['target']
x=tr.drop(['ID_code','target'],axis=1,errors='ignore')


x.info()


from sklearn.model_selection import train_test_split as splt
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier as lgbm
import lightgbm

xtr,xval,ytr,yval = splt(x,y,stratify = y,random_state=42, test_size=0.2)


lgb = lgbm(
    objective='binary',
    metrics='auc',
    n_estimators=1000,
    learning_rate = 0.05,
    random_state=42,
    n_jobs=-1
)

lgb.fit(
    xtr,ytr,
    eval_set = [(xval,yval)],
    eval_metric = 'auc',
    callbacks=[lightgbm.early_stopping(stopping_rounds=100, verbose=False)]
)


yval_pr = lgb.predict_proba(xval)[:,1]

val_auc = roc_auc_score(yval,yval_pr)
print(f"최종 검증 ROC AUC: {val_auc:.6f}")


xte = te.drop('ID_code',axis=1,errors='ignore')
te_pr=lgb.predict_proba(xte)[:,1]

submission = pd.DataFrame({'ID_code':te['ID_code'], 'target':te_pr})
submission.to_csv('submission.csv', index=False)

