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


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv',index_col='id')
train.head()


y = train['accident_risk']
X = train.drop('accident_risk',axis=1)


X.columns


X.head()


def convert_category(dataframe):
    for col in dataframe:
        dataframe[col] = dataframe[col].astype('category') 


convert_category(X)



test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


convert_category(test)


X.info()


test.info()


train[train.duplicated()]

train = train.drop_duplicates()


test_ids = test['id']

test = test.drop('id',axis=1)


from sklearn.model_selection import KFold
import lightgbm as lgb
from lightgbm import LGBMRegressor 
from sklearn.metrics import mean_squared_error
test_preds = np.zeros(len(test))
oof_preds_lgbm = np.zeros(len(X)) 

kf = KFold(n_splits=5,random_state=42,shuffle=True) 
model = LGBMRegressor(n_estimators=10000,
                     learning_rate = 0.03,
                     num_leaves = 31,
                     random_state= 42)

for train_idx,val_idx in kf.split(X,y):
    X_train,X_val = X.iloc[train_idx],X.iloc[val_idx]
    y_train,y_val = y.iloc[train_idx],y.iloc[val_idx]
    model.fit(X_train,y_train)
    oof_preds_lgbm[val_idx] = model.predict(X_val)
    test_preds += model.predict(test)/kf.n_splits 
rmse = np.sqrt(mean_squared_error(y,oof_preds_lgbm)) 
print(rmse)

pd.DataFrame({'light_gbm_oof':oof_preds_lgbm,'target':y}).to_csv('lgbm_oof.csv',index=False)



final = pd.DataFrame({'id':test_ids,
              'accident_rate':test_preds  
})

final.to_csv('submission.csv',index=False)

























