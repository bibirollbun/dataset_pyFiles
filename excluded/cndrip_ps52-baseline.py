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
import seaborn as sns


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


train.describe()





# 查看缺失值情况
train.isna().sum().sort_values(ascending=False)


train['Price'].skew(),train['Price'].kurtosis()


sns.histplot(data=train['Price'], kde=True, stat="density");


from sklearn.preprocessing import LabelEncoder


train = train.drop('id', axis = 1)
test = test.drop('id', axis = 1)
num_cols = list(test.select_dtypes(exclude=['object']).columns.difference(['num_sold']))
cat_cols = list(test.select_dtypes(include=['object']).columns)




num_cols,cat_cols


num_cols


for col in cat_cols:
    train[col].fillna('Unknown', inplace=True)
    test[col].fillna('Unknown', inplace=True)

# 对分类数据进行编码
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le  # 保存LabelEncoder对象以便将来可能需要逆变换

# 处理数值数据的缺失值
for col in num_cols:
    mean_value = train[col].mean()
    train[col].fillna(mean_value, inplace=True)
    test[col].fillna(mean_value, inplace=True)


train.info()


from sklearn.model_selection import train_test_split
X = train.drop(['Price'], axis=1)
y = train['Price']


# Split datainto training set and test set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_error
# Define RMSE metric
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))



rfr = RandomForestRegressor(random_state=42)
rfr.fit(X_train, y_train)
# 预测和评估
y_pred = rfr.predict(X_valid)

score = rmse(y_valid, y_pred)
score


xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)
# 预测和评估
y_pred = xgb.predict(X_valid)

score = rmse(y_valid, y_pred)
score


lgb = LGBMRegressor(random_state=42)
lgb.fit(X_train, y_train)
# 预测和评估
y_pred = lgb.predict(X_valid)

score = rmse(y_valid, y_pred)
score


cat = CatBoostRegressor(random_state=42)
cat.fit(X_train, y_train)
# 预测和评估
y_pred = cat.predict(X_valid)

score = rmse(y_valid, y_pred)
score


xgb_pred=xgb.predict(test)


rfr_pred=rfr.predict(test)


lgb_pred=lgb.predict(test)


cat_pred=cat.predict(test)


submission  = pd.read_csv(r"/kaggle/input/playground-series-s5e2/sample_submission.csv")


submission['Price']=xgb_pred
# submission.to_csv('submission_xgb.csv', index=False)
submission.head()


submission['Price']=lgb_pred
# submission.to_csv('submission_lgb.csv', index=False)
submission.head()


(rfr_pred,xgb_pred,lgb_pred,cat_pred)





submission['Price']=np.mean([rfr_pred,xgb_pred,lgb_pred,cat_pred],axis=0)
submission.to_csv('submission_all.csv', index=False)
submission.head()

