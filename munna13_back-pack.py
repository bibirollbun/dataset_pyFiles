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

train=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train


train_extra=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
train_extra


train_data=pd.concat([train,train_extra],ignore_index=True)
train_data


train_data.isnull().sum()


#Imputation 
categorical_cols=train_data.select_dtypes(include=['object']).columns
numerical_cols=train_data.select_dtypes(include=['int64','float64']).columns


for col in categorical_cols:
  train_data[col].fillna(train_data[col].mode()[0],inplace=True)


for col in numerical_cols:
  train_data[col].fillna(train_data[col].median(),inplace=True)




from sklearn.preprocessing import OrdinalEncoder 


encoder=OrdinalEncoder()
train_data[categorical_cols]=encoder.fit_transform(train_data[categorical_cols])
train_data


x=train_data.drop('Price',axis=1)
y=train_data['Price']




from sklearn.model_selection import train_test_split

X_train,X_val,Y_train,Y_val=train_test_split(x,y,test_size=0.3,random_state=42)




from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score


rf=RandomForestRegressor(n_estimators=50)
rf.fit(X_train,Y_train)
y_pred=rf.predict(X_val)




test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')



#Imputation
categorical_cols=test_df.select_dtypes(include=['object']).columns
numerical_cols=test_df.select_dtypes(include=['int64','float64']).columns


for col in categorical_cols:
  test_df[col].fillna(test_df[col].mode()[0],inplace=True)


for col in numerical_cols:
  test_df[col].fillna(test_df[col].median(),inplace=True)





test_df[categorical_cols] = encoder.transform(test_df[categorical_cols])


test_df['Price'] = rf.predict(test_df)


test_df = test_df.reset_index()
test_df


d=test_df[['id','Price']]
d.to_csv('submission.csv',index=False)

