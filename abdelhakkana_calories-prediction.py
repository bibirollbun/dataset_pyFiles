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


X_train,X_test,y_train,y_test=train_test_split(X2,y,test_size=0.04,random_state=44)


from sklearn.linear_model import LinearRegression


r1=LinearRegression()
r1.fit(X_train,y_train)
y1_pred=r1.predict(X_test)
y1_pred=np.sqrt(np.power(y1_pred,2))
np.sqrt(mean_squared_log_error(y_test,y1_pred))


test


testt=test[X.columns]


testt


testty=st.transform(testt)


predictions=r1.predict(testty)


predictions=np.sqrt(np.power(predictions,2))
predictions


output=pd.DataFrame({'id':test.id,'Calories':predictions})
output


submission=output.to_csv('submission.csv',index=False)




