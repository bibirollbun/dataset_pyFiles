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


train=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample=pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


train.head()


test.head()


train.shape



test.shape



train.info()


train.describe()


for column in train.columns:
    print(train[column].isnull().value_counts())


for column in train.columns:
    print(column in test.columns)
    if column not in test.columns:
        print(column)


for column in train.columns:
    if train[column].isnull().any():
        print(column)


for column in train.select_dtypes(include=['object']):
    print(train[column].value_counts())


def sort_columns(column):
    column=column.str.extract(r'(\d+)')
    column=column.astype(int)

    return column


train['Episode_Title']=sort_columns(train['Episode_Title'])


test['Episode_Title']=sort_columns(test['Episode_Title'])


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def onehot_encoding(column):
    ohe=OneHotEncoder()
    ohe.fit_transform(column.astype(str))

    return column


def label_encoding(column):
    le=LabelEncoder()
    column=le.fit_transform(column.astype(str))

    return column


for column in train.select_dtypes(include=['object']):
    train[column]=label_encoding(train[column])


for column in test.select_dtypes(include=['object']):
    test[column]=label_encoding(test[column])


train.info()


train.head()


import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge


train=train.drop('id',axis=1)


train.head()


test=test.drop('id',axis=1)


test.head()


train.info()


y=train['Listening_Time_minutes']


X=train
X=X.drop('Listening_Time_minutes',axis=1)


X.shape


train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=2025)


train_x


model1=xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators= 5000,
    max_depth= 14,
    learning_rate= 0.0337948950616333,
    subsample= 0.8340874974199307,
    colsample_bytree= 0.7846367146534168,
    enable_categorical=True,
    device = "cuda",
    random_state=2025
)



# params = {
#     'learning_rate': 0.1,
#     'n_estimators': 1000,
#     'max_depth': 14,
#     'num_leaves': 200,
#     'min_child_samples': 69,
#     'subsample': 0.7332671514523149,
#     'colsample_bytree': 0.6877349185747408,
#     'reg_alpha': 0.02088233530065522,
#     'reg_lambda': 0.012008381265393863,
#     'verbose': 1,
#     'device': 'gpu',  # 使用GPU加速
#     'objective': 'regression',  # 回归任务（可省略，默认值）
#     'random_state': 2025  # 建议添加随机种子
# }


model2=lgb.LGBMRegressor(
        n_iter=1500,
        max_depth=-1,
        num_leaves=1024,
        colsample_bytree=0.7,
        learning_rate=0.04,   #0.03,0.05
        objective='l2',
        metric='rmse', 
        verbosity=-1,
        max_bin=1024,
        random_state=2025,
)


model3=BayesianRidge(
    n_iter=100,
    alpha_1=1e-6,    # 超参数α的先验参数（控制权重稀疏性）
    alpha_2=1e-6,    # 超参数α的先验参数
    lambda_1=1e-6,   # 超参数λ的先验参数（控制噪声精度）
    lambda_2=1e-6    # 超参数λ的先验参数
)


model1.fit(train_x,train_y)


model2.fit(train_x,train_y)


train_y_pred_1=model1.predict(train_x)


train_y_pred_2=model2.predict(train_x)


train_meta= pd.DataFrame({'xgb':train_y_pred_1,'lgb':train_y_pred_2})


model3.fit(train_meta,train_y)





test_pred_1=model1.predict(test_x)


test_pred_2=model2.predict(test_x)


test_meta= pd.DataFrame({'xgb':test_pred_1,'lgb':test_pred_2})


test_pred_3=model3.predict(test_meta)


print(mean_squared_error(test_y,test_pred_3,squared=False)) 


model1.fit(X,y)


model2.fit(X,y)


y_pred_1=model1.predict(X)


y_pred_2=model2.predict(X)


train_meta=pd.DataFrame({'xgb':y_pred_1,'lgb':y_pred_2})


train_meta.head()





model3.fit(train_meta,y)


test_y_pred_1=model1.predict(test)


test_y_pred_2=model2.predict(test)


test_meta=pd.DataFrame({'xgb':test_y_pred_1,'lgb':test_y_pred_2})


meta_test=model3.predict(test_meta)


sample.head()


sample['Listening_Time_minutes']=meta_test


sample


sample.to_csv('submissionv1-stack.csv',index=False)

