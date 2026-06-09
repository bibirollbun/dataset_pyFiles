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
import matplotlib.pyplot as plt
import seaborn as sns
df_train=pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv",index_col="id")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv",index_col="id")
print(df_train.head())
X=df_train.copy()
y=X.pop("Calories")


from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import  RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


for col in X.select_dtypes("object"):
    X[col],_=X[col].factorize()
    df_test[col],_=df_test[col].factorize()
print(df_test.head(5))


from sklearn.model_selection import KFold
kf=KFold(n_splits=3)
X_fold=X.copy()
X_fold["kfold"]=-1
for fold,(tr,val) in enumerate(kf.split(X_fold)):
    X_fold.loc[val,"kfold"]=fold
X_fold



def rmlse(yvalid,pred):
    return np.mean(np.sqrt((np.log1p(yvalid)-np.log1p(np.abs(pred)))**2))
lgbm=LGBMRegressor(random_state=42,n_estimators=600,reg_lambda=0.8,learning_rate=0.3,n_jobs=-1,force_col_wise=True)
cbr=CatBoostRegressor(random_state=42,n_estimators=600,reg_lambda=0.8,learning_rate=0.3)
xgb=XGBRegressor(random_state=42,n_estimators=600,reg_lambda=0.8,learning_rate=0.3,n_jobs=-1)
model=[lgbm,cbr,xgb]


scores=[]
for i in range(3):
    print(f"training{i}-------------")
    xtrain=X[X_fold["kfold"]!=i]
    xvalid=X[X_fold["kfold"]==i]

    ytrain=y[X_fold["kfold"]!=i]
    yvalid=y[X_fold["kfold"]==i]

    model[i].fit(xtrain,ytrain)
    pred=model[i].predict(xvalid)
    scores.append(rmlse(yvalid,pred))
print(scores)
np.mean(scores)


total=np.sum(scores)
model_weight=[scores[0]/total,scores[1]/total,scores[2]/total]
model_weight


pred_1=model[0].predict(df_test)
pred_2=model[1].predict(df_test)
pred_3=model[2].predict(df_test)
pred_total=pred_1*model_weight[0]+pred_2*model_weight[1]+pred_3*model_weight[2]


sample_sub=pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
print(sample_sub.shape)
sample_sub["Calories"]=(np.abs(pred_total))
sample_sub.tail
sample_sub.to_csv("model_blending_pred_2.csv",index=False)

