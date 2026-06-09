# Python environment for this notebook is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_data=pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_data=pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


train_data.head()


train_data.shape


train_data.info()


train_data.describe()


train_data.isnull().sum()


test_data.isna().sum()


train_data.hist(['BeatsPerMinute'],bins=10)
plt.show()


X=train_data.drop(columns=['id','BeatsPerMinute'],axis=1)
y=train_data['BeatsPerMinute']
scaler = MinMaxScaler()
X= scaler.fit_transform(X)
test_features = scaler.transform(test_data.iloc[:,1:])


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=21)


from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(random_state=21, max_depth=9)


rfr.fit(X_train,y_train)


rfr.feature_importances_


y_hat=rfr.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_hat, y_test))
print(f'the root mean squared error on validation set is: {rmse}')


y_predict=rfr.predict(test_features)


submission = pd.DataFrame({"id": test_data['id']})
#submission['BeatsPerMinute']=train_data.describe().loc['mean']['BeatsPerMinute']
submission['BeatsPerMinute']=y_predict


submission.to_csv('submission.csv', index=False)




