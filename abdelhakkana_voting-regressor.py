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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_log_error


train=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train


X=train.drop(['id','Sex','Calories'],axis=1)
y=train['Calories']


sc=StandardScaler()
st=MinMaxScaler()


X1=sc.fit_transform(X)
X2=st.fit_transform(X)


X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=0.04,random_state=44)


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor,RandomForestRegressor,VotingRegressor,StackingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor 
from xgboost import XGBRegressor


r1=LinearRegression()
r1.fit(X_train,y_train)
y1_pred=r1.predict(X_test)
y1_pred=np.sqrt(np.power(y1_pred,2))
np.sqrt(mean_squared_log_error(y_test,y1_pred))


r2=HistGradientBoostingRegressor(random_state=33)
r2.fit(X_train,y_train)
y2_pred=r2.predict(X_test)
y2_pred=np.sqrt(np.power(y2_pred,2))
np.sqrt(mean_squared_log_error(y_test,y2_pred))


r3=RandomForestRegressor(random_state=11,n_jobs=-1)
r3.fit(X_train,y_train)
y3_pred=r3.predict(X_test)
y3_pred=np.sqrt(np.power(y3_pred,2))
np.sqrt(mean_squared_log_error(y_test,y3_pred))


r4=LGBMRegressor(random_state=1,verbose=0)
r4.fit(X_train,y_train)
y4_pred=r4.predict(X_test)
y4_pred=np.sqrt(np.power(y4_pred,2))
np.sqrt(mean_squared_log_error(y_test,y4_pred))


r5=XGBRegressor(random_state=83,n_jobs=-1)
r5.fit(X_train,y_train)
y5_pred=r5.predict(X_test)
y5_pred=np.sqrt(np.power(y5_pred,2))
np.sqrt(mean_squared_log_error(y_test,y5_pred))


r6=CatBoostRegressor(random_state=22,verbose=0)
r6.fit(X_train,y_train)
y6_pred=r6.predict(X_test)
y6_pred=np.sqrt(np.power(y6_pred,2))
np.sqrt(mean_squared_log_error(y_test,y6_pred))


voting=VotingRegressor(estimators=[('rf',r3),('cb',r6)])
voting.fit(X_train,y_train)
y_pred=voting.predict(X_test)
y_pred=np.sqrt(np.power(y_pred,2))
np.sqrt(mean_squared_log_error(y_test,y_pred))


test


testt=test[X.columns]


testt


testty=sc.transform(testt)


predictions=voting.predict(testty)


predictions=np.sqrt(np.power(predictions,2))
predictions


output=pd.DataFrame({'id':test.id,'Calories':predictions})
output


submission=output.to_csv('submission.csv',index=False)




